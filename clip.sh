#!/bin/bash

# 基础配置
DATA_DIR="/root/datasets/Market-1501"  
MODEL="clip_vit_b16"                  
LOG_DIR="logs/market-1501/CLIP_fin_new"      


CUDA_VISIBLE_DEVICES=0,1 python PCB.py \
  -d market \
  -a "$MODEL" \
  -b 64 \
  -j 4 \
  --epochs 60 \
  --lr 3e-4 \
  --features 512 \
  --height 224 \
  --width 224 \
  --step-size 15 \
  --dropout 0.2 \
  --logs-dir "$LOG_DIR" \
  --data-dir "$DATA_DIR"