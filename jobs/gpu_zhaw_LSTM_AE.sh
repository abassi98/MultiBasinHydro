#!/bin/bash

#SBATCH --job-name=gpu_LSTM_AE_bdTrue30_1Y
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM_AE-bdTrue30_1Y.out
#SBATCH --error=gpu_LSTM_AE-bdTrue30_1Y.err

module load gcc/9.4.0-pe5.34 miniconda3/4.12.0 lsfm-init-miniconda/1.0.0	
conda activate my_env

python3 src/LSTM_AE_main.py --num_features 30 --bidirectional 1 --debug 0 # bidirectional, training mode
