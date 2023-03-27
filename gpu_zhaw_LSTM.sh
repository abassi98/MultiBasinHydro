#!/bin/bash

#SBATCH --job-name=gpu_LSTM_Stat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM-bdTrueStat.out
#SBATCH --error=gpu_LSTM-bdTrueStat.err

 USS/2020
module load USS/2020 gcc/7.3.0 miniconda3/4.8.2 lsfm-init-miniconda/1.0.0

python3 src/LSTM_main.py --noise_dim 0 --statics 1 --bidirectional 1 --debug 0 
