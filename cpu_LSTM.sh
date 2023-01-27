#!/bin/bash
#SBATCH -J cpu_LSTM_AE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # 8 cores
#SBATCH --time=120:00:00                      # 120-hours run-time
#SBATCH --mem-per-cpu=1G            # 1  GB per core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL
#SBATCH --output=cpu_LSTM_AE.out
#SBATCH --error=cpu_LSTM_AE.err

module load gcc/8.2.0 python_gpu/3.10.4    # Load modules      
python3 src/MultiBasinHydro_lupoalberto98/LSTM_main.py          # Execute the program
