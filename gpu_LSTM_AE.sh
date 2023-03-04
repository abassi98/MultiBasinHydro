#!/bin/bash -l

<<<<<<< HEAD
#SBATCH --job-name=gpu_LSTM_AE_bdTrue3
=======
#SBATCH --job-name=gpu_LSTM_AE_bdTrue4
>>>>>>> 651ba154828a4ea3cdd71f231b79421b96fb78bc
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --account=em09      
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
<<<<<<< HEAD
#SBATCH --output=gpu_LSTM_AE_bdTrue3.out
#SBATCH --error=gpu_LSTM_AE_bdTrue3.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu PyTorch     
python3 src/MultiBasinHydro_lupoalberto98/LSTM_AE_main.py --num_features 3 --bidirectional 1 --debug 0 # no bidirectional, training mode
=======
#SBATCH --output=gpu_LSTM_AE_bdTrue4.out
#SBATCH --error=gpu_LSTM_AE_bdTrue4.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu PyTorch     
python3 src/MultiBasinHydro_lupoalberto98/LSTM_AE_main.py --num_features 4 --bidirectional 1 --debug 0 # no bidirectional, training mode
>>>>>>> 651ba154828a4ea3cdd71f231b79421b96fb78bc
