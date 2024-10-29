import numpy as np
import time
import dask

from dask.distributed import Client


def inc(x):
	time.sleep(1)
	print(x + 1)
	return 0




def main():
	client = Client(threads_per_worker = 1, n_workers=4)	
	data = [1, 2, 3, 4]
	output = []
	
	print('Direct simulation starting')
	start = time.perf_counter()
	for x in data:
		output.append(inc(x))
	stop = time.perf_counter()
	print('Direct simulation finished in ', stop - start, ' seconds ')

	print('Dask simulation starting')
	@dask.delayed
	def daskinc(x):
		time.sleep(1)
		print(x + 1)
		return 1

	start = time.perf_counter()
	for x in data:
		output.append(daskinc(x))

	output = dask.persist(*output)
	stop = time.perf_counter()
	print('Dask simulation finished in ', stop - start, ' seconds ')

if __name__ == '__main__':
	main()
