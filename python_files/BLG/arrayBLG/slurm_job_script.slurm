#!/bin/bash
#SBATCH --job-name=grid_calculation
#SBATCH --output=results_%A_%a.out
#SBATCH --error=results_%A_%a.err
#SBATCH --array=0-59%20  # 60 total workers, max 20 running simultaneously
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00

# Load necessary modules (adjust as needed for your environment)
module load python/3.8

# Activate virtual environment if using one
# source /path/to/your/venv/bin/activate

# Run the Python script
python grid_calculation.py

# Check if this is the last array task
if [ "$SLURM_ARRAY_TASK_ID" -eq $((SLURM_ARRAY_TASK_MAX)) ]; then
    echo "This is the last task. Running post-processing script."
    python results_combiner.py
fi
