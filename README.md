## Official implementation code of DISCO.AHU team
![image](https://github.com/user-attachments/assets/4dc4af1f-a8a7-4979-9e42-fc3dfed1a731)



## 🔥 Winning 2nd Place in WWW2025 Multimodal CTR Prediction Challenge Track

### 🔥 Follow to perfectly reproduce the results of this code.

- To facilitate reproducibility, we share the model checkpoints on Hugging Face: https://huggingface.co/salmon1802/QIN/tree/main
- In ./checkpoints and ./submission folders have our run logs and submission files, respectively.
- This submission can be reproduced manually by following the actions below, or by directly using the one-click run script run.sh

### Data Preparation

1. Download the datasets at: https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR

2. Unzip the data files to the `data` directory

    ```bash
   cd ./data/
   wget -r -np -nH --cut-dirs=1 http://recsys.westlake.edu.cn/MicroLens_1M_MMCTR/MicroLens_1M_x1/
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
The parameters QIN_variety_v9 in __./config/qin_config/model_config.yaml__ are set to the optimal hyperparameters in the environment described above.

#### Tips
It is worth mentioning that after our tests, we find that although the parameter num_row = 4 achieves the best performance in the above environments, there is training instability in some environments.

When this happens, we suggest that sacrificing some performance in favor of setting num_row = 3 reproduces the results well.

## Citation
If you find our code helpful for your research, please cite the following paper:

```bibtex
@article{li2025quadratic,
  title={Quadratic Interest Network for Multimodal Click-Through Rate Prediction},
  author={Li, Honghao and Li, Hanwei and Zhang, Jing and Zhang, Yi and Yu, Ziniu and Sang, Lei and Zhang, Yiwen},
  journal={arXiv preprint arXiv:2504.17699},
  year={2025}
}
```
