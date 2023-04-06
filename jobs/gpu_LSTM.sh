#!/bin/bash -l

#SBATCH --job-name=gpu_LSTM-bdTrueStat   
#SBATCH --time=24:00:00          
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --account=em09
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM-bdTrueStat.out
#SBATCH --error=gpu_LSTM-bdTrueStat.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu PyTorch      
python3 ../src/LSTM_main.py --noise_dim 0 --statics 1 --bidirectional 1 --debug 0 # no noise, static features addes, bidirectional, training mode