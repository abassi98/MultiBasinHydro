#!/bin/bash
#SBATCH --job-name="gpu_LSTM_AE"        # create a short name for your job
#SBATCH --gpus=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=2G                 # memory per gpu-core (4G is default)
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --output="LSTM_AE_gpu.out"
#SBATCH --error="LSTM_AE_gpu.err"

module load gcc/8.2.0 python_gpu/3.10.4    # Load modules      
python src/MultiBasinHydro_lupoalberto98/LSTM_AE_main.py          # Execute the program