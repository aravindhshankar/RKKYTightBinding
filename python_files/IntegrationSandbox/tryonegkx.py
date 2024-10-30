import sys 
sys.path.insert(1,'../')

import os 
import numpy as np
from scipy.integrate import simpson, quad
from matplotlib import pyplot as plt
from h5_handler import * 
from scipy.signal import find_peaks



path_to_loadfile = '../Dump/BLG/AliceGkxomegaSolveRGF/'

if not os.path.exists(path_to_loadfile): 
	print("path to dump directory not found - create data first!")
	exit(1)

def esimps():
	omega = 0.00039 
	filename = f'BLG_Gkx_omega_{omega:.5f}.h5'
	print(filename)
	if not os.path.exists(os.path.join(path_to_loadfile, filename)):
		print('LOADFILE DOES NOT EXIST!')
		exit(1)

	loadpath = os.path.join(path_to_loadfile, filename)
	load_dict = h52dict(loadpath, verbose=True)
	fig, ax = plt.subplots(1)
	
	Gkx = load_dict['Gkx']
	kxvals = load_dict['kxvals']
	omegaval = load_dict['omega']
	print(load_dict.keys())
	ax.plot(kxvals, Gkx, '.-')	
	ax.set_title(f'omega = {omegaval:.5f}')
	simpsval = simpson(Gkx, kxvals)
	print('Simpson integration done with value ', simpsval)
	plt.show()





if __name__ == '__main__':
	esimps()
