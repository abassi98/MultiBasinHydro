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

module load gcc/7.3.0
module load cuda/11.6.2

python3 src/LSTM_main.py --noise_dim 0 --statics 1 --bidirectional 1 --debug 0 
