#!/bin/bash -l

#SBATCH --job-name=gpu_LSTM_AE_nf5      
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --account=em09      
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM_AE_nf5.out
#SBATCH --error=gpu_LSTM_AE_nf5.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu PyTorch     
python3 src/MultiBasinHydro_lupoalberto98/LSTM_AE_nf5_main.py         