import numpy as np
import pandas as pd
import os
import shutil
from grid_config import rvals, omegavals

def combine_results():
    # Find all result chunk files in tmp directory
    result_files = [f for f in os.listdir('tmp') if f.startswith('results_chunk_') and f.endswith('.npy')]
    
    # Sort files to ensure correct order
    result_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    # Load and combine results
    full_results = []
    for file in result_files:
        chunk = np.load(os.path.join('tmp', file))
        full_results.append(chunk)
    
    # Concatenate results along the omega axis
    final_results = np.concatenate(full_results, axis=2)
    
    # Create separate DataFrames for each of the 4 output values
    output_dataframes = {}
    output_names = ['f1', 'f2', 'f3', 'f4']  # Modify these names as needed
    
    for i, name in enumerate(output_names):
        df = pd.DataFrame(
            final_results[i, :, :], 
            index=[f'r_{r:.4f}' for r in rvals],  # Row labels
            columns=[f'omega_{omega:.4f}' for omega in omegavals]  # Column labels
        )
        df.to_csv(f'{name}_complete_grid_results.csv')
        output_dataframes[name] = df
    
    print(f"Combined results saved. Shape of each output: {final_results.shape[1:]}")
    
    # # Optional: remove the tmp directory after combining
    # shutil.rmtree('tmp')
    # print("Temporary results directory removed.")

if __name__ == '__main__':
    combine_results()
