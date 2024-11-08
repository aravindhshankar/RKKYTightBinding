from __future__ import absolute_import, unicode_literals, print_function
import os
# virtualenv_lib_path = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/.BLGvenv/lib/cubalib'
# os.environ['DYLD_LIBRARY_PATH'] = f"{virtualenv_lib_path}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"  # macOS
import pycuba
#pycuba.demo()
import numpy as np
import math

def print_header(name):
	print('-------------------- %s test -------------------' % name)
def print_results(name, results):
	keys = ['nregions', 'neval', 'fail']
	text = ["%s %d" % (k, results[k]) for k in keys if k in results]
	print("%s RESULT:\t" % name.upper() + "\t".join(text))
	for comp in results['results']:
		print("%s RESULT:\t" % name.upper() + \
			"%(integral).8f +- %(error).8f\tp = %(prob).3f\n" % comp)

LOWERLIM = -np.pi
UPPERLIM = np.pi
LOWERLIM = 0
UPPERLIM = 1
def shiftdomain(f):
	'''
	f(x) is integrated from a to b
	Output: g(y) is integrated from 0 to 1 to give the same result
	'''

	return lambda y: (UPPERLIM-LOWERLIM) * f(LOWERLIM + (UPPERLIM-LOWERLIM)*y) 

def cubify(f):
	'''
	Wrapper that takes a function of one variable which needs to be integrated from 0,1 and returns a cuba style Integrand
	'''
	def wrapper(ndim,xx,ncomp,ff,userdata):
		x,_ = [xx[i] for i in range(ndim.contents.value)]
		ff[0] = (UPPERLIM-LOWERLIM) * f(LOWERLIM + (UPPERLIM-LOWERLIM)*x)
		return 0
	return wrapper 


if __name__ == '__main__':
	@cubify
	def twopeakIntegrand(x):
		a = 1e-5
		pref = 1 / (a * np.pi) 
		peaklist = np.sort([0.12312, 0.12413, 0.13123, 0.43535, 0.657575,0.12312,0.34535,0.25363,0.536344,0.235235,0.24353,0.78878,0.89898435,0.3423,0.567,0.4333])
		# result = pref * 1 / ( 1 + ((x-0.1312)**2 / a**2)) + pref * 1 / ( 1 + ((x-0.73217)**2 / a**2))
		result = np.sum([pref * 1 / ( 1 + ((x-x0)**2 / a**2)) for x0 in peaklist]) 
		return result

	NDIM = 2
	NCOMP = 1

	NNEW = 1000
	NMIN = 100
	# FLATNESS = 50.
	FLATNESS = 1e-3

	# KEY1 = 47
	KEY1 = 47
	KEY2 = 1
	KEY3 = 1
	MAXPASS = 50000
	BORDER = 0.
	MAXCHISQ = 0.01
	MINDEVIATION = .025
	NGIVEN = 0
	LDXGIVEN = NDIM
	NEXTRA = 0
	MINEVAL = 0
	MAXEVAL = 500000

	# Integrand = sharpIntegrand
	Integrand = twopeakIntegrand
	KEY = 0
	verbose=0

	print_header('Vegas')
	print_results('Vegas', pycuba.Vegas(Integrand, NDIM, verbose=verbose, maxeval=MAXEVAL,  epsrel=1e-3))

	print_header('Suave')
	print_results('Suave', pycuba.Suave(Integrand, NDIM, NNEW, NMIN, FLATNESS, verbose=verbose,maxeval = MAXEVAL,  epsrel=1e-3))

	# print_header('Divonne')
	# print_results('Divonne', pycuba.Divonne(Integrand, NDIM, 
				# mineval=MINEVAL, maxeval=MAXEVAL,
				# key1=KEY1, key2=KEY2, key3=KEY3, maxpass=MAXPASS,
				# border=BORDER, maxchisq=MAXCHISQ, mindeviation=MINDEVIATION,
				# ldxgiven=LDXGIVEN, verbose=verbose,epsrel=1e-3))

	print_header('Cuhre')
	print_results('Cuhre', pycuba.Cuhre(Integrand, NDIM, key=KEY, maxeval=MAXEVAL, verbose=verbose,epsrel=1e-6 ))

