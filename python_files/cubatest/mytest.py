from __future__ import absolute_import, unicode_literals, print_function
import os
# virtualenv_lib_path = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/.BLGvenv/lib/cubalib'
# os.environ['DYLD_LIBRARY_PATH'] = f"{virtualenv_lib_path}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"  # macOS
import pycuba
#pycuba.demo()
import numpy as np

def print_header(name):
	print('-------------------- %s test -------------------' % name)
def print_results(name, results):
	keys = ['nregions', 'neval', 'fail']
	text = ["%s %d" % (k, results[k]) for k in keys if k in results]
	print("%s RESULT:\t" % name.upper() + "\t".join(text))
	for comp in results['results']:
		print("%s RESULT:\t" % name.upper() + \
			"%(integral).8f +- %(error).8f\tp = %(prob).3f\n" % comp)

def cubify(f,a,b):
	'''
	f(x) is integrated from a to b
	g(y) is integrated from 0 to 1 to give the same result
	'''

	return lambda x: (b-a) * f(a + (b-a)*y) 
 


if __name__ == '__main__':
	import math

	def Integrand(ndim, xx, ncomp, ff, userdata):
		x,y,z = [xx[i] for i in range(ndim.contents.value)]
		result = np.exp(-x**2)
		ff[0] = result
		return 0
	def sharpIntegrand(ndim, xx, ncomp, ff, userdata):
		x,_ = [xx[i] for i in range(ndim.contents.value)]
		a = 1e-5
		pref = 1 / (a * np.pi) 
		result = pref * 1 / ( 1 + ((x-0.1312)**2 / a**2))
		ff[0] = result
		return 0
	def twopeakIntegrand(ndim, xx, ncomp, ff, userdata):
		x,_ = [xx[i] for i in range(ndim.contents.value)]
		# x = xx[0]
		a = 1e-5
		pref = 1 / (a * np.pi) 
		result = pref * 1 / ( 1 + ((x-0.1312)**2 / a**2)) + pref * 1 / ( 1 + ((x-0.73217)**2 / a**2))
 
		# ff[0] = result * np.sin(np.pi * y) * 0.5 * np.pi
		ff[0] = result 
		return 0

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
	MAXEVAL = 1000000

	# Integrand = sharpIntegrand
	Integrand = twopeakIntegrand
	KEY = 0
	verbose=0

	print_header('Vegas')
	print_results('Vegas', pycuba.Vegas(Integrand, NDIM, verbose=verbose, epsrel=1e-3))

	print_header('Suave')
	print_results('Suave', pycuba.Suave(Integrand, NDIM, NNEW, NMIN, FLATNESS, verbose=verbose, epsrel=1e-3))

	print_header('Divonne')
	print_results('Divonne', pycuba.Divonne(Integrand, NDIM, 
				mineval=MINEVAL, maxeval=MAXEVAL,
				key1=KEY1, key2=KEY2, key3=KEY3, maxpass=MAXPASS,
				border=BORDER, maxchisq=MAXCHISQ, mindeviation=MINDEVIATION,
				ldxgiven=LDXGIVEN, verbose=verbose,epsrel=1e-3))

	print_header('Cuhre')
	print_results('Cuhre', pycuba.Cuhre(Integrand, NDIM, key=KEY, verbose=verbose,epsrel=1e-6 ))

