#!/bin/bash -l

#SBATCH --job-name=analysis_basins       
#SBATCH --time=24:00:00          
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=em09
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=analsyis_basins.out
#SBATCH --error=analsyis_basins.err
#SBATCH --constraint=mc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu PyTorch 
pip3 install scikit-learn     
python3 src/MultiBasinHydro_lupoalberto98/analysis_basins.py --model_id "lstm-ae" --best_epoch 3779        