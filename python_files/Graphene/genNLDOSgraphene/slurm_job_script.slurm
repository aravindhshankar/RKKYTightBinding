#!/bin/bash
#SBATCH --job-name=grid-calculation
#SBATCH --array=0-399
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=2-23:00:00
#SBATCH --output=logs/%A_%a.out     # Standard output log file (%A: job ID, %a: array index)
#SBATCH --error=logs/%A_%a.err      # Standard error log file
#SBATCH --partition=cpu-long      # Partition (queue) name
#SBATCH --mail-type=ALL             # Notifications for job completion or failure
# SBATCH --mail-user=shankar@lorentz.leidenuniv.nl # Email to send notifications (optional)
#SBATCH --mail-user=ashankar@ictp.it # Email to send notifications (optional)

# Create the logs/ directory if it doesn't exist
mkdir -p logs

# Print job details for logging purposes
echo "Starting job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID"
echo "Running on host $(hostname)"
echo "Using $SLURM_CPUS_PER_TASK CPUs"
echo "Memory per task: $SLURM_MEM_PER_NODE"
echo "Working directory: $(pwd)"

# Load necessary modules (adjust as needed for your environment)
module load Python/3.10.8-GCCcore-12.2.0
#
source $HOME/BLGvenv/bin/activate
#
# Run the Python script
# python grid_calculation.py
python grid-calculation.py

# Check if this is the last array task
# if [ "$SLURM_ARRAY_TASK_ID" -eq $((SLURM_ARRAY_TASK_MAX)) ]; then
#     echo "This is the last task. Running post-processing script."
#     python results_combiner.py
# fi
echo "SLURM JOB SCRIPT ENDED" 
echo "NOW RUN results_combiner.py"
echo "............exiting.............."
