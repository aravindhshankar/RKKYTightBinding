#!/bin/bash
#SBATCH --job-name=v4BLG_LDOS_11
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="shankar@lorentz.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-4:00:00
#SBATCH --partition=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=24
##SBATCH --exclusive
#SBATCH --cpus-per-task=1
#
module load Python/3.10.8-GCCcore-12.2.0
#
source $HOME/.venv/bin/activate
#
echo "[$SHELL] ## This is $SLURM_JOB_USER on #HOSTNAME and job ID $SLURM_JOB_ID"
export CWD=$(pwd)
echo "[$SHELL] Current working directory: $CWD"
#
export file_name="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
echo "[$SHELL] Running Python script"
#
python v4_RGF_BBG.py "$file_name"
#
echo "[$SHELL]#### SLURM script finished. "
