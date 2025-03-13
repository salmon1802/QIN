import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.utils import not_in_whitelist
import logging

class QIN_variety_v9(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="QIN_variety_v9",
                 gpu=-1,
                 num_layers=3,
                 num_row=2,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 factor=0.1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(QIN_variety_v9, self).__init__(feature_map,
                                             model_id=model_id,
                                             gpu=gpu,
                                             embedding_regularizer=embedding_regularizer,
                                             net_regularizer=net_regularizer,
                                             **kwargs)
        self.feature_map = feature_map
        self.factor = factor
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.attention_layers = MultiHeadTargetAttention(input_dim=self.item_info_dim,
                                                         attention_dim=self.item_info_dim * 2)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim
        self.qnn = QuadraticNeuralNetworks(input_dim=input_dim,
                                           num_layers=num_layers,
                                           net_dropout=net_dropout,
                                           num_row=num_row,
                                           batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict:  # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]
        pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)
        emb_list += [target_emb, pooling_emb]
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.qnn(feature_emb)
        return_dict = {"y_pred": self.output_activation(y_pred)}
        return return_dict

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay(factor=self.factor)
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

class MultiHeadTargetAttention(nn.Module):
    def __init__(self,
                 input_dim=64,
                 attention_dim=64,
                 use_scale=True):
        super(MultiHeadTargetAttention, self).__init__()
        self.attention_dim = attention_dim
        self.scale = self.attention_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_o = nn.Linear(attention_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention() # remove dropout 96.81

    def forward(self, target_item, history_sequence, mask=None):
        """
        target_item: (B, D)
        history_sequence: (B, seq_len, D)
        mask: (B, seq_len)
        """
        query = self.W_q(target_item)            # (B, attention_dim)
        key = self.W_k(history_sequence)         # (B, seq_len, attention_dim)
        value = self.W_v(history_sequence)       # (B, seq_len, attention_dim)
        batch_size = query.size(0)
        query = query.view(batch_size, 1, self.attention_dim)     # (B, 1, attention_dim)
        key = key.view(batch_size, -1, self.attention_dim)        # (B, seq_len, attention_dim)
        value = value.view(batch_size, -1, self.attention_dim)    # (B, seq_len, attention_dim)

        if mask is not None:
            mask = mask.view(batch_size, 1, -1)
        output = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        output = output.view(batch_size, -1)
        output = self.W_o(output) + target_item
        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores * mask
        attention = self.relu(scores)
        output = torch.matmul(attention, V)
        return output

class QuadraticNeuralNetworks(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers=3,
                 net_dropout=0.1,
                 num_row=2,
                 batch_norm=False):
        super(QuadraticNeuralNetworks, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.layer = nn.ModuleList()
        self.activation = nn.ModuleList()
        for i in range(num_layers):
            self.layer.append(QuadraticLayer(input_dim, num_row=num_row, net_dropout=net_dropout))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(input_dim))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            self.activation.append(nn.PReLU())
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        for i in range(self.num_layers):
            residual = x
            x = self.layer[i](x)
            if len(self.norm) > i:
                x = self.norm[i](x)
            if self.activation[i] is not None:
                x = self.activation[i](x)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
            x = x + residual
        logit = self.fc(x)
        return logit

class QuadraticLayer(nn.Module):
    def __init__(self, input_dim, num_row=2, net_dropout=0.1):
        super(QuadraticLayer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim * num_row),
                                    nn.Dropout(net_dropout))
        self.num_row = num_row
        self.input_dim = input_dim

    def forward(self, x):  # Khatri-Rao Product
        h = self.linear(x).view(-1, self.num_row, self.input_dim)  # B × R × D
        x = torch.einsum("bd,brd->bd", x, h)
        return x