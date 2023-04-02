#!/bin/bash

#SBATCH --job-name=gpu_LSTM_StatL2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM-bdTrueStatL2.out
#SBATCH --error=gpu_LSTM-bdTrueStatL2.err

module load gcc/9.4.0-pe5.34 miniconda3/4.12.0 lsfm-init-miniconda/1.0.0	
conda activate my_env
python3 src/LSTM_main.py --noise_dim 0 --statics 1 --hydro 1 --debug 0 
