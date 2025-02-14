# grid_calculation.py
import numpy as np
import os
import sys
from grid_config import rvals, omegavals, NUMGS
from grid_config import helper_mp as f
from matplotlib import pyplot as plt

def get_chunk_boundaries(array_length, total_workers):
    """
    Calculate chunk boundaries with smaller chunks in the middle and larger chunks at the ends,
    using a Gaussian-based distribution to match computation time differences.
    Middle:End ratio is approximately 1:12 to balance workload.
    """
    # Create points centered around the middle
    x = np.linspace(-2, 2, total_workers)
    
    # Create inverse Gaussian distribution (smaller in middle, larger at ends)
    # The 2.0 coefficient controls the ratio between largest and smallest chunks
    weights = 1.0 + 2.0 * np.exp(-(x**2))
    
    # Flip the weights (we want smaller values in the middle)
    weights = weights.max() - weights + weights.min()
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Calculate cumulative chunk sizes
    cumulative_sizes = (weights * array_length).cumsum()
    
    # Convert to integer indices
    indices = np.round(cumulative_sizes).astype(int)
    indices = np.insert(indices, 0, 0)  # Ensure starting at 0
    indices[-1] = array_length  # Ensure ending at array_length
    
    return indices

def demonstrate_chunk_sizes(array_length=1600, total_workers=100):
    """
    Demonstrate the chunk sizes for given parameters
    """
    boundaries = get_chunk_boundaries(array_length, total_workers)
    chunk_sizes = np.diff(boundaries)
    
    # Print some key statistics
    print(f"\nChunk size distribution for {array_length} elements across {total_workers} workers:")
    print(f"Largest chunk size: {chunk_sizes.max()}")
    print(f"Smallest chunk size: {chunk_sizes.min()}")
    print(f"Average chunk size: {chunk_sizes.mean():.1f}")
    print(f"Ratio (largest:smallest): {chunk_sizes.max()/chunk_sizes.min():.1f}")
    
    # Print sizes for first 5, middle 5, and last 5 chunks
    print("\nFirst 5 chunks:")
    for i in range(5):
        print(f"Worker {i}: {chunk_sizes[i]} elements")
    
    print("\nMiddle 5 chunks:")
    mid = total_workers // 2
    for i in range(mid-2, mid+3):
        print(f"Worker {i}: {chunk_sizes[i]} elements")
    
    print("\nLast 5 chunks:")
    for i in range(total_workers-5, total_workers):
        print(f"Worker {i}: {chunk_sizes[i]} elements")
    
    # Estimate processing times
    end_time_per_element = 10/chunk_sizes[0]  # seconds
    print("\nEstimated processing times:")
    print(f"First chunk: {end_time_per_element * chunk_sizes[0]:.1f} seconds")
    print(f"Middle chunk: {end_time_per_element * chunk_sizes[mid]:.1f} seconds")
    print(f"Last chunk: {end_time_per_element * chunk_sizes[-1]:.1f} seconds")
    
    task_array = np.arange(total_workers)
    starts = [boundaries[task_id] for task_id in task_array] 
    omegastarts = [omegavals[start] for start in starts]
    fig, ax = plt.subplots(1)
    # ax.bar(x = starts, height = chunk_sizes, width=chunk_sizes-1)
    ax.semilogx(omegastarts,chunk_sizes,'p')
    print(f'length of starts = {len(starts)}')
    print(starts)
    print(f'last stop = {boundaries[total_workers]}')
    plt.show()

def main():
    # Create tmp directory if it doesn't exist
    os.makedirs('tmp', exist_ok=True)

    # Get the array task ID and total number of workers from Slurm environment variables
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    total_workers = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    # Get chunk boundaries for all workers
    chunk_boundaries = get_chunk_boundaries(len(omegavals), total_workers)
    
    # Determine the start and end indices for this worker
    start_idx = chunk_boundaries[task_id]
    end_idx = chunk_boundaries[task_id + 1]
    
    # Select the subset of omega values for this worker
    worker_omegavals = omegavals[start_idx:end_idx]
    
    # Perform the grid calculation (NUMGS result values for each grid point)
    results = np.zeros((NUMGS, len(rvals), len(worker_omegavals)))
    for i, r in enumerate(rvals):
        for j, omega in enumerate(worker_omegavals):
            results[:, i, j] = f(r, omega)
    
    # Save results to a file in tmp directory
    output_filename = os.path.join('tmp', f'results_chunk_{task_id}.npy')
    np.save(output_filename, results)
    
    chunk_size = end_idx - start_idx
    print(f"Processed chunk {task_id}: {start_idx} to {end_idx} (size: {chunk_size}, Total workers: {total_workers})")

if __name__ == '__main__':
    # demonstrate_chunk_sizes(len(omegavals), 100)
    main()
