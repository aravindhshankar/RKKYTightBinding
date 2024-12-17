# grid_calculation.py
import numpy as np
import os
import sys
from grid_config import rvals, omegavals

def f(r, omega):
    """
    Your grid calculation function.
    Replace this with your actual implementation.
    """
    # Example placeholder function - replace with your actual calculation
    return r * np.sin(omega)

def main():
    # Create tmp directory if it doesn't exist
    os.makedirs('tmp', exist_ok=True)

    # Get the array task ID and total number of workers from Slurm environment variables
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    total_workers = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 60))
    
    # Calculate the chunk size for omega values
    chunk_size = len(omegavals) // total_workers
    
    # Determine the start and end indices for this worker
    start_idx = task_id * chunk_size
    end_idx = start_idx + chunk_size if task_id < total_workers - 1 else len(omegavals)
    
    # Select the subset of omega values for this worker
    worker_omegavals = omegavals[start_idx:end_idx]
    
    # Perform the grid calculation
    results = np.zeros((len(rvals), len(worker_omegavals)))
    for i, r in enumerate(rvals):
        for j, omega in enumerate(worker_omegavals):
            results[i, j] = f(r, omega)
    
    # Save results to a file in tmp directory
    output_filename = os.path.join('tmp', f'results_chunk_{task_id}.npy')
    np.save(output_filename, results)
    
    print(f"Processed chunk {task_id}: {start_idx} to {end_idx} (Total workers: {total_workers})")

if __name__ == '__main__':
    main()
