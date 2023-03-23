#!/bin/bash -l

#SBATCH --job-name=gpu_LSTM-bdTrueStat   
#SBATCH --time=24:00:00          
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --partition=earth-4
#SBATCH --account=xbs4  
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM-bdTrueStat.out
#SBATCH --error=gpu_LSTM-bdTrueStat.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load python/3.9.12-pe5.34 #module load daint-gpu PyTorch      
python3 src/LSTM_main.py --noise_dim 0 --statics 1 --bidirectional 1 --debug 0 # no noise, static features addes, bidirectional, training mode