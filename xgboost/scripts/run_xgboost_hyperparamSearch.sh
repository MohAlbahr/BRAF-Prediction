#!/bin/bash

# set visible GPUs
export CUDA_VISIBLE_DEVICES=1
 
# try to avoid WandB timeout issues
export WANDB__SERVICE_WAIT=300


# use miniconda, fix problems with tmux
eval "$('/homes/malbahri/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"


echo "Here is the code from: xgboost_hyperparamSearch.py"
# FineTuning
conda run -n provPath --live-stream python xgboost_hyperparamSearch.py