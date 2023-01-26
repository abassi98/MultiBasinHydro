
#!/bin/bash
#SBATCH -J cpu_LSTM_AE
#SBATCH -n 8                    # 8 cores
#SBATCH -t 120:00                   # 12-hours run-time
#SBATCH --mem-per-cpu=1G            # 1  GB per core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL
#SBATCH --output=cpu_LSTM_AE.out
#SBATCH --error=cpu_LSTM_AE.err

module load gcc/8.2.0 python_gpu/3.10.4    # Load modules      
python src/MultiBasinHydro_lupoalberto98/LSTM_AE_main.py          # Execute the program
