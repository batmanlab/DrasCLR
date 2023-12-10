#!/usr/bin/env bash

set -x

LOG_DIR="./logs/"
EXPERIMENT="exp_neighbor_2_128_expert_8"

mkdir -p ${LOG_DIR}

python train.py \
  --exp-name=${EXPERIMENT} \
  --root-dir='/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/patch_data_32_6_reg_mask/' \
  --world-size=1 \
  --rank=0 \
  --dist-url='tcp://localhost:10032' \
  --dist-backend='nccl' \
  --npgus-per-node=4 \
  --workers=16 \
  --epochs=20 \
  --start-epoch=0 \
  --resume='' \
  --print-freq=10 \
  --seed=0 \
  --num-patch=581 \
  --batch-size=128 \
  --lr=0.01 \
  --rep-dim=128 \
  --moco-dim=128 \
  --moco-k=4096 \
  --moco-m=0.999 \
  --moco-t=0.2 \
  --num-experts=8 \
  --warm-up=0 \
  --k-neighbors=2 \
  --beta=1.0 \
  --augmentation='agc'
