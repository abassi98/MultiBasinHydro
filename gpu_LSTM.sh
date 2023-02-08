#!/bin/bash
#SBATCH --job-name=gpu_LSTM       # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpumem:4G                 # memory per gpu-core (4G is default)
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM.out
#SBATCH --error=gpu_LSTM.err
#SBATCH --account=gp0002

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load daint-gpu PyTorch      
python3 src/MultiBasinHydro_lupoalberto98/LSTM_main.py          # Execute the program