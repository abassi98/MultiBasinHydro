#!/bin/bash -l

#SBATCH --job-name=analysis       
#SBATCH --time=24:00:00          
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --account=em09
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=analsyis.out
#SBATCH --error=analsyis.err
#SBATCH --constraint=mc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu PyTorch      
python3 src/MultiBasinHydro_lupoalberto98/analysis.py        