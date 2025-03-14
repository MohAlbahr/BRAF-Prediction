#!/bin/bash

# Export environment variables
export CUDA_VISIBLE_DEVICES=3
export HF_TOKEN=''
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%,}

# Task setting
TASKCFG=/prov-gigapath/finetune/task_configs/braf.yaml

## for the whole TCGA dataset only
DATASETCSV=/prov-gigapath/dataset_csv/braf/tcga_tiles_embeddings_with_labels.csv
PRESPLITDIR=/prov-gigapath/dataset_csv/braf/braf_5_folds/braf # test contains here the a subset of TCGA Dataset file names (as usual in cross validation)
SPLITDIR=/prov-gigapath/dataset_csv/braf/braf_5_folds
MAX_WSI_SIZE=250000  # Maximum WSI size in pixels for the longer side (width or height).

TILE_SIZE=256
# Model settings
HFMODEL=hf_hub:prov-gigapath/prov-gigapath # Huggingface model name
MODELARCH=gigapath_slide_enc12l768d
TILEEMBEDSIZE=1536
LATENTDIM=768
# Training settings
EPOCH=5
GC=32
BLR=0.003
WD=0.05
LD=0.95
FEATLAYER="9"
DROPOUT=0.5
DROPPATHRATE=0.1
# Output settings
WORKSPACE=/prov-gigapath/outputs/braf
SAVEDIR=$WORKSPACE


PROVTYPE=unfreeze
TRAINTYPE="predict_only"


EXPNAME=run_traintype-${TRAINTYPE}_epoch-${EPOCH}_blr-${BLR}_BS-${GC}_wd-${WD}_ld-${LD}_drop-${DROPOUT}_dropPR-${DROPPATHRATE}_feat-${FEATLAYER}


# Try to avoid WandB timeout issues
export WANDB__SERVICE_WAIT=300

# Use miniconda, fix problems with tmux
eval "$(conda shell.bash hook)"

# Activate the Conda environment
source /homes/malbahri/miniconda3/etc/profile.d/conda.sh
conda activate provPath
export CUDA_LAUNCH_BLOCKING=1

# Remove permissions for other users
umask 007

mkdir -p /local/work/tcga_uke_tiles_embeddings/
rsync --copy-links --recursive --times --group --no-perms --chmod=ugo=rwX --verbose -P "/prov-gigapath/data/tcga_uke_tiles_embeddings/" "/local/work/tcga_uke_tiles_embeddings/"

ROOTPATH=/local/work/tcga_uke_tiles_embeddings
echo "Data directory set to $ROOTPATH"

# Fine-tuning
python /prov-gigapath/finetune/predict_CV_on_tcga.py \
                                                           --task_cfg_path ${TASKCFG} \
                                                           --dataset_csv $DATASETCSV \
                                                           --root_path $ROOTPATH \
                                                           --model_arch $MODELARCH \
                                                           --blr $BLR \
                                                           --layer_decay $LD \
                                                           --optim_wd $WD \
                                                           --dropout $DROPOUT \
                                                           --drop_path_rate $DROPPATHRATE \
                                                           --val_r 0.1 \
                                                           --epochs $EPOCH \
                                                           --input_dim $TILEEMBEDSIZE \
                                                           --latent_dim $LATENTDIM \
                                                           --feat_layer $FEATLAYER \
                                                           --warmup_epochs 1 \
                                                           --gc $GC \
                                                           --model_select last_epoch \
                                                           --lr_scheduler cosine \
                                                           --folds 1 \
                                                           --save_dir $SAVEDIR \
                                                           --pretrained $HFMODEL \
                                                           --report_to wandb \
                                                           --exp_name $EXPNAME \
                                                           --max_wsi_size $MAX_WSI_SIZE \
                                                           --split_dir $SPLITDIR \
                                                           --pre_split_dir $PRESPLITDIR
