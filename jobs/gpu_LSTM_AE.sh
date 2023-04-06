#!/bin/bash -l

#SBATCH --job-name=gpu_LSTM_AE_bdTrue3
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --account=em09      
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM_AE_bdTrue3.out
#SBATCH --error=gpu_LSTM_AE_bdTrue3.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu PyTorch     
python3 ../src/LSTM_AE_main.py --num_features 3 --bidirectional 1 --debug 0 # no bidirectional, training mode
