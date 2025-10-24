#!/bin/bash

# 设置训练环境变量
export PYTHONPATH="/fs/scratch/PAS2473/MM2025/CVPR2026/mntp_zero/mntp:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0  # 设置使用的GPU

# 单节点训练
torchrun --nproc_per_node=1 --master_port=29502 /fs/scratch/PAS2473/MM2025/CVPR2026/mntp_zero/mntp/train.py \
  --depth=16 \
  --bs=40 \
  --ep=40 \
  --fp16=1 \
  --tblr=1e-4 \
  --alng=1e-3 \
  --wpe=0.1 \
  --data_load_reso=256 \