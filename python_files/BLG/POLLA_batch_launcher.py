import numpy as np
import os
from itertools import product


#######################
### User parameters ###
#######################

# name of the script to be run (must be in parent directory)
script = "2-run_npfabric-vqe.py"
# job name will be created from this, inserting parameter values
jobname_template = "npf-L{}N{}U{}i{}d{}"

param_iterators = (
    [8],  # L (nsites)
    [4],  # N (nelec)
    [4.],  # U (coulomb)
    np.arange(1, 4),  # d (depth)
    np.arange(0, 1000),  # i (index)
)

time = "1-00:00:00"  # format days-hh:mm
mem = "1GB"  # can use postfixes (MB, GB, ...)
partition = "ibIntel"  # "compIntel"

# insert here additional lines that should be run before the script
# (source bash scripts, load modules, activate environment, etc.)
from additional_lines import additional_lines

current_dir = os.getcwd()
job_dir = os.path.join(*os.path.split(current_dir), 'jobs')
out_dir = os.path.join(*os.path.split(current_dir), 'out')
err_dir = os.path.join(*os.path.split(current_dir), 'err')

os.chdir('..')

os.makedirs(job_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
os.makedirs(err_dir, exist_ok=True)

for L, N, U, d, i in product(*param_iterators):
    job_name = jobname_template.format(L, N, U, i, d)
    job_file = os.path.join(job_dir, job_name + '.job')

    with open(job_file, 'wt') as fh:
        fh.writelines(
            ["#!/bin/bash\n",
             f"#SBATCH --job-name={job_name}\n",
             f"#SBATCH --output={os.path.join(out_dir, job_name+'.out')}\n",
             f"#SBATCH --error={os.path.join(err_dir, job_name+'.err')}\n",
             f"#SBATCH --time={time}\n",
             f"#SBATCH --mem={mem}\n",
             f"#SBATCH --partition={partition}\n",
             f"#SBATCH --mail-type=NONE\n",
             ] + additional_lines + [
                f"python -u {script} {L} {N} {U} {i} {d}\n"]
        )

    os.system("sbatch " + job_file)
