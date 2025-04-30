#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python PCB.py \
  -d market \
  -a vit_small \
  --lr 0.01 \
  --features 256 \
  --height 384 \
  --width 128 \
  --batch-size 64 \
  --workers 4 \
  --epochs 160 \
  --step-size 25 \
  --data-dir ~/datasets/Market-1501 \
  --logs-dir logs/market-1501/DINO_fin