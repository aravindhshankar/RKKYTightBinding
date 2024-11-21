#!/bin/bash
#SBATCH --job-name=NLDOSblg    # Job name
#SBATCH --array=0-20                # Job array range: one task per r value
#SBATCH --ntasks=1                  # Number of tasks per job (1 because we use multi-threading, not MPI)
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=4G                    # Memory allocation per task
#SBATCH --time=12:00:00             # Maximum runtime per task
#SBATCH --output=logs/%A_%a.out     # Standard output log file (%A: job ID, %a: array index)
#SBATCH --error=logs/%A_%a.err      # Standard error log file
#SBATCH --partition=cpu-medium      # Partition (queue) name
#SBATCH --mail-type=ALL             # Notifications for job completion or failure
#SBATCH --mail-user=shankar@lorentz.leidenuniv.nl # Email to send notifications (optional)

# Create the logs/ directory if it doesn't exist
mkdir -p logs

# Print job details for logging purposes
echo "Starting job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID"
echo "Running on host $(hostname)"
echo "Using $SLURM_CPUS_PER_TASK CPUs"
echo "Memory per task: $SLURM_MEM_PER_NODE"
echo "Working directory: $(pwd)"

# Load necessary modules (adjust based on your system)
module load Python/3.10.8-GCCcore-12.2.0
#
source $HOME/BLGvenv/bin/activate
#
# Run the Python script with the array task ID as an argument
python blgNLDOS.py $SLURM_ARRAY_TASK_ID
#
# Print completion message
echo "Job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID completed at $(date)"
