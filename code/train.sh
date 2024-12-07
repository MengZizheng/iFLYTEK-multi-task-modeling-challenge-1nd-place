#!/bin/bash

# 运行 preprocess.py 文件
python ./preprocess.py

# 切换到 user_data 文件夹下的 LLaMA-Factory 目录
cd ../user_data/user/LLaMA-Factory

# 运行训练命令
llamafactory-cli train examples/train_lora/qwen2_5_lora_sft.yaml

# 导出模型
llamafactory-cli export examples/merge_lora/qwen2_5_lora_sft.yaml

# 返回到 code 文件夹
cd ../../code

# 运行 train.py 文件
python ./train.py