#!/bin/bash

#SBATCH -N 1
#SBATCH -c 16
#SBATCH -p general
#SBATCH -q public
#SBATCH --gpus=a100:1
#SBATCH --mem=20G
#SBATCH -t 3-00:00:00
#SBATCH -o log_folder/slurm.%j.out
#SBATCH -e log_folder/slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rpaul12@asu.edu
module load mamba
source activate cmb_seg
module load cuda-11.8.0-gcc-12.1.0

python -u main_se.py --method cnn_classifier --paired_dataset ALL --model_save_path ./saved_models_dice_wce_UNetClassifier/ --batch_size 1 --lr 0.001 --n_workers 4
