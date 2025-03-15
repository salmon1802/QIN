#!/bin/bash

# 初始化 Conda 环境
eval "$(conda shell.bash hook)"

# 创建 data 目录（如果不存在）
mkdir -p data

# 检查 data 目录是否为空，如果为空则下载数据
if [ -z "$(ls -A data)" ]; then
    echo "正在下载数据..."
    cd data
    wget -r -np -nH --cut-dirs=1 http://recsys.westlake.edu.cn/MicroLens_1M_MMCTR/
    python combine_item.py
    cd ..
else
    echo "data 目录非空，跳过数据下载。"
fi

# 检查并创建 Conda 环境（如果不存在）
if ! conda env list | grep -q fuxictr_www; then
    echo "正在创建 Conda 环境 fuxictr_www..."
    conda create -n fuxictr_www python==3.8 -y
fi

# 安装所需的依赖包
echo "正在安装依赖包..."
conda run -n fuxictr_www --no-capture-output pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
conda run -n fuxictr_www --no-capture-output pip install -r requirements.txt

# 运行实验
echo "正在运行实验..."
conda run -n fuxictr_www --no-capture-output python run_expid.py

echo "完成。"