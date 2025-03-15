## WWW2025_MMCTR_Challenge

## 🔥 Follow to perfectly reproduce the results of this code.

##### In . /checkpoints and . /submission folders have our run logs and submission files, respectively.

#### This submission can be reproduced manually by following the actions below, or by directly using the one-click run script run.sh

### Data Preparation

1. Download the datasets at: https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR

2. Unzip the data files to the `data` directory

    ```bash
   cd ./data/
   wget -r -np -nH --cut-dirs=1 http://recsys.westlake.edu.cn/MicroLens_1M_MMCTR/
    ```
   
3. Processing likes_level and views_level features into sequence features

    ```bash
   cd ./data/
   python combine_item.py
    ```

### Environment

We run the experiments on RTX 4090 GPU of AutoDL.com

Please set up the environment as follows. 

+ torch==2.0.0+cu118
+ fuxictr==2.3.7

   ```
   conda create -n fuxictr_www python==3.8
   pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   source activate fuxictr_www
   ```

### How to Run

Train the model on train and validation sets:

    ```
    python run_expid.py
    ```

The parameters in __./config/qin_config/__ are set to the optimal hyperparameters in the environment described above.