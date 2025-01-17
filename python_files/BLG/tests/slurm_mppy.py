import sys
import os
import numpy as np
import multiprocessing as mp
import time
from dask.distributed import Client

def helper_fn(x):
	# print(x)
	time.sleep(2)
	return x**2

def pooler():
	xvals = np.arange(64)
	PROCESSES = mp.cpu_count()
	# PROCESSES = int(os.environ['SLURM_CPUS_PER_TASK'])

	start_time = time.perf_counter()
	with mp.Pool(PROCESSES) as pool:
		squares = pool.map(helper_fn,xvals)

	elapsed = time.perf_counter() - start_time
	print(f'Parallel computation of {len(xvals)} elements with {PROCESSES} processes finished in time {elapsed} seconds')
	print(xvals, squares)


def dask_client():
	xvals = np.arange(64)
	PROCESSES = mp.cpu_count()
	client = Client(threads_per_worker=1, n_workers=PROCESSES)

	start_time = time.perf_counter()
	squares = client.gather(client.map(helper_fn,xvals))

	elapsed = time.perf_counter() - start_time
	print(f'Parallel computation of {len(xvals)} elements with {PROCESSES} processes finished in time {elapsed} seconds')
	print(xvals, squares)




if __name__ == '__main__':
	# dask_client()
	pooler()
	## works perfectly
