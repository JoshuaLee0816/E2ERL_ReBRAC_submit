#!/bin/bash
# Run Q-landscape comparison (beta=0 vs beta=0.001, epoch 500)
#
# Before running:
#   Place critic_params.msgpack in:
#     checkpoints/beta0/
#     checkpoints/beta0001/
#
# Output saved to outputs/q_landscape_compare.png and .pdf

cd "$(dirname "$0")"

python q_landscape_compare.py \
    --beta0_ckpt  checkpoints/beta0    \
    --beta1_ckpt  checkpoints/beta0001 \
    --beta0_val   0.0                  \
    --beta1_val   0.001                \
    --dataset     maze2d-umaze-v1      \
    --output      outputs/q_landscape_compare.png
