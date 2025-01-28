#!/bin/bash
#SBATCH --job-name=dispatch-bt
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out


apptainer run --nv \
    /home/myasincifci/containers/main/main.sif \
    python ./train.py --config-name vaelstm_moving_mnist