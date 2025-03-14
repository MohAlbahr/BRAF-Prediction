#!/bin/bash

source /homes/malbahri/miniconda3/etc/profile.d/conda.sh  # the path where Conda is initialized
conda activate provPath
export CUDA_LAUNCH_BLOCKING=1

# remove permissions for other users
umask 007

# copy datasets
# mkdir -p /local/work/TCGA-SKCM/

mkdir -p /local/work/TCGA-SKCM/
mkdir -p /local/work/tiles_tcga/
rsync --copy-links --recursive --times --group --no-perms --chmod=ugo=rwX --verbose -P "/projects/wispermed/TCGA/" "/local/work/TCGA-SKCM/"

# mask_path="/local/work/TCGA-SKCM"
python3 create_tiles_dataset.py
