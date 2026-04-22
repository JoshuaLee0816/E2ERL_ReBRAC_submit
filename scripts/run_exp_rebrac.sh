#!/bin/bash

# =============================================================================
# USER CONFIG — change these to match your environment before running
# =============================================================================
MUJOCO_DIR="/path/to/your/mujoco210" 
CONDA_ENV_BIN="/path/to/your/conda/bin"
PROJECT_ROOT="/path/to/your/projectroot"    
D4RL_DATA="/path/to/your/d4rl_datasets"        
GPU_ID=1                                                        
# =============================================================================

export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_DIR}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_DIR}/bin:/usr/lib/nvidia
export PATH=$PATH:/usr/bin:/usr/local/cuda/bin
export D4RL_DATASET_DIR=${D4RL_DATA}
export WANDB_MODE=online

PYTHON_BIN=${CONDA_ENV_BIN}/python
REBRAC_PY=${PROJECT_ROOT}/algorithms/offline/rebrac.py
CONFIG_BASE=${PROJECT_ROOT}/configs/offline/rebrac

# Hyperparameters settings
SEED=(1 42 80 100)
BETA=(0.005 0.01)
PROJECT_NAME="E2ERL(ReBRAC)"

ENVS=(
    # --- MuJoCo Locomotion ---
    "halfcheetah-medium-v2"        "halfcheetah-medium-replay-v2"  "halfcheetah-medium-expert-v2"
    "hopper-medium-v2"             "hopper-medium-replay-v2"       "hopper-medium-expert-v2"
    "walker2d-medium-v2"           "walker2d-medium-replay-v2"     "walker2d-medium-expert-v2"
    # --- Maze2D ---
    "maze2d-umaze-v1"              "maze2d-medium-v1"              "maze2d-large-v1"
    # --- AntMaze ---
    "antmaze-umaze-v2"             "antmaze-umaze-diverse-v2"
    "antmaze-medium-play-v2"       "antmaze-medium-diverse-v2"
    "antmaze-large-play-v2"        "antmaze-large-diverse-v2"
    # --- Adroit ---
    "pen-human-v1"                 "pen-cloned-v1"                 "pen-expert-v1"
    "door-human-v1"                "door-cloned-v1"                "door-expert-v1"
    "hammer-human-v1"              "hammer-cloned-v1"              "hammer-expert-v1"
    "relocate-human-v1"            "relocate-cloned-v1"            "relocate-expert-v1"
)

# Main Loop
for B in "${BETA[@]}"
do
    for ENV in "${ENVS[@]}"
    do
        for S in "${SEED[@]}"
        do
            ENV_FOLDER=$(echo "$ENV" | cut -d'-' -f1)
            YAML_FILE=$(echo "$ENV" | cut -d'-' -f2- | tr '-' '_').yaml
            CONFIG_PATH="${CONFIG_BASE}/${ENV_FOLDER}/${YAML_FILE}"

            if [ ! -f "$CONFIG_PATH" ]; then
                echo "Can't find config: $CONFIG_PATH,Skip $ENV"
                continue
            fi

            echo "============================================"
            echo "Run: $ENV | E2E_beta: $B | Seed: $S"
            echo "Time: $(date +'%H:%M:%S')"
            echo "============================================"

            # --- run ---
            CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_BIN "$REBRAC_PY" \
                --config_path "$CONFIG_PATH" \
                --train_seed $S \
                --project "$PROJECT_NAME" \
                --group "${ENV}-B${B}" \
                --name "s${S}" \
                --E2E_beta $B \

            if [ $? -eq 0 ]; then
                echo "$ENV Training Completed Successfully."
            else
                echo "$ENV E2E_beta-$B Seed-$S Failed" >> "${PROJECT_ROOT}/error_log.txt"
            fi
        done
    done
done

echo "All Training Completed."
