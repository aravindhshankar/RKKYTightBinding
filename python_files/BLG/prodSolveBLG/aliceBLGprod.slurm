#!/bin/bash
#SBATCH --job-name=Om6CUHREBLGsolveRGF
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="shankar@lorentz.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-22:00:00
#SBATCH --partition=cpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=8
##SBATCH --exclusive
#SBATCH --cpus-per-task=1
#
module load Python/3.10.8-GCCcore-12.2.0
#
source $HOME/BLGvenv/bin/activate
#
echo "[$SHELL] ## This is $SLURM_JOB_USER on #HOSTNAME and job ID $SLURM_JOB_ID"
export CWD=$(pwd)
echo "[$SHELL] Current working directory: $CWD"
#
export file_name="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
echo "[$SHELL] Running Python script"
#
python CUHREcomputeLDOS.py "$file_name"
#
echo "[$SHELL]#### SLURM script finished. "
