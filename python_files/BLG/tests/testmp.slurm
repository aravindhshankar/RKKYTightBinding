#!/bin/bash
#SBATCH --job-name=testmp
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="shankar@lorentz.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=100M
#SBATCH --time=0-0:10:00
#SBATCH --partition=compAMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
python slurm_mppy.py
#
echo "[$SHELL]#### SLURM script finished. "
