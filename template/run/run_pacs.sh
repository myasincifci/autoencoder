#!/bin/bash
#SBATCH --job-name=dispatch-bt
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/dispatch_smol/data/PACS.hdf5 /tmp

ls /tmp

apptainer run --nv \
    /home/myasincifci/containers/main/main.sif \
    python ./train.py --config-name base