import sys 
import os 
import numpy as np 
import mpmath as mpm
sys.path.insert(0,'..')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks 
from scipy.integrate import simpson, quad
from functools import partial, cache
import time 
from FastRGF.RGF import MOMfastrecDOSfull
from h5_handler import *



def func_with_sing(x):
	f1 = 1./np.sqrt(np.abs(x-1.))
	f2 = np.log(np.abs(x-1.))
	return f1


def simple():
	''' What did we learn? 
	'''
	# intval = quad(func_with_sing,0,2)
	# print(intval)
	# print("done")
	intval1 = quad(func_with_sing,0,1)
	intval2 = quad(func_with_sing,1,2)
	print(f'intval1 = {intval1}')
	print(f'intval2 = {intval2}')
	integ = intval1[0]+intval2[0]
	print(f'Actual exact integral = {integ}')
	intwithbreakpoints = quad(func_with_sing,0,2,points=1.0001)
	print(f'Int with breakpoints = {intwithbreakpoints}')



##### Now test integrating just the singularities in graphene/BLG - check how much precision is needed in the breakpoint

if __name__ == '__main__':
	simple()