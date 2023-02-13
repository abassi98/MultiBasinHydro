#!/bin/bash -l

#SBATCH --job-name=extract_features      
#SBATCH --time=24:00:00          
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=em09
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=extract_features.out
#SBATCH --error=extract_features.err
#SBATCH --constraint=mc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu PyTorch 
#pip3 install scikit-learn seaborn 
python3 src/MultiBasinHydro_lupoalberto98/extract_features.py