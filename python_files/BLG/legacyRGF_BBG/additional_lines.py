ENV_PATH = '/home/polla/git_repos/DFTQML/.conda'

additional_lines = [
    "module purge\n",
    "ml load GCC/13.2.0\n", # fixes an error caused by matplotlib requiring GCC
    "module load QuantumMiniconda3/4.7.10 \n",
    "source activate " + ENV_PATH  + " \n",
]
