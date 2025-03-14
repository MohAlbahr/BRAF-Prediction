#!/bin/bash

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=1,2
export WANDB_MODE=offline

# Try to avoid WandB timeout issues
export WANDB__SERVICE_WAIT=300
export HF_TOKEN=''  # put HF_TOKEN for Prov-GigaPath !

# Use miniconda, fix problems with tmux
eval "$('/homes/malbahri/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the relative path to the Python script and the save directory
PYTHON_SCRIPT="$SCRIPT_DIR/../braf-provpath/generate_tile_embeddings.py"
SAVE_DIR="$SCRIPT_DIR/../braf-provpath/data/tcga_uke_tiles_embeddings"

# Run the Python script using the specified conda environment
conda run -n provPath --live-stream python "$PYTHON_SCRIPT" --config_name prov_config --datasets 'tcga' --run_number 1 --save_dir "$SAVE_DIR"
