#!/bin/bash
#PBS -N ce_training
#PBS -l walltime=6:00:00
#PBS -q gpu@pbs-m1.metacentrum.cz 
#PBS -l select=1:ncpus=4:mem=3gb:ngpus=1:gpu_mem=20gb
#PBS -m ae
#PBS -j oe

# source conda init 
source ~/.bashrc

module add mambaforge
mamba activate /storage/brno12-cerit/home/xjarol06/mamba/md2d

HOMEDIR=/storage/brno12-cerit/home/xjarol06/multidoc2dial-reranking

echo "$PBS_JOBID is running on node `hostname -f`" >> $HOMEDIR/jobs_info.txt

cd $HOMEDIR

# Set the bash variables for each option
NUM_EPOCHS=30
STOP_TIME="5h"
LR=1e-5
WEIGHT_DECAY=1e-1
DROPOUT_RATE=0.1
LABEL_SMOOTHING=0
GRADIENT_CLIP=5
SAVE_MODEL_PATH="ce_lr${LR}_wd${WEIGHT_DECAY}_dr${DROPOUT_RATE}_ls${LABEL_SMOOTHING}_gc${GRADIENT_CLIP}.pt"

echo "Model will be saved to $SAVE_MODEL_PATH"
echo "Starting training"

# Construct the command with the parameters
python main.py --train \
    --num_epochs $NUM_EPOCHS \
    --stop_time $STOP_TIME \
    --save_model_path $SAVE_MODEL_PATH \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --dropout_rate $DROPOUT_RATE \
    --label_smoothing $LABEL_SMOOTHING \
    --gradient_clip $GRADIENT_CLIP




