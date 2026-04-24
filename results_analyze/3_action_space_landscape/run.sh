#!/bin/bash
# Run action-space Q landscape comparison (beta=0, 0.001, 0.005)
#
# Before running:
#   Place critic_params.msgpack in:
#     checkpoints/beta0/
#     checkpoints/beta0001/
#     checkpoints/beta0005/
#
# Output saved to outputs/action_space_compare.png and .pdf

cd "$(dirname "$0")"

python action_space_viz.py \
    --beta0_ckpt  checkpoints/beta0    \
    --beta1_ckpt  checkpoints/beta0001 \
    --beta2_ckpt  checkpoints/beta0005 \
    --beta0_val   0.0                  \
    --beta1_val   0.001                \
    --beta2_val   0.005                \
    --anchor_x    1.0                  \
    --anchor_y    1.0                  \
    --radius      0.005                \
    --output      outputs/action_space_compare.png
