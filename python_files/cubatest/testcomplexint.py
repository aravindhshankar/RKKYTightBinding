########################### LEARNING : INTEGRATE THE REAL AND IMAGINARY PARTS SEPARATELY #########################
from __future__ import absolute_import, unicode_literals, print_function
import os
import pycuba
import numpy as np
import math
import sys
sys.path.insert(1,'..')
from utils.decorators import cubify

##### initialize cubify ######
cubify.set_limits(-np.pi,np.pi)

def print_header(name):
	print('-------------------- %s test -------------------' % name)
def print_results(name, results):
	keys = ['nregions', 'neval', 'fail']
	text = ["%s %d" % (k, results[k]) for k in keys if k in results]
	print("%s RESULT:\t" % name.upper() + "\t".join(text))
	for comp in results['results']:
		print("%s RESULT:\t" % name.upper() + \
			"%(integral).8f +- %(error).8f\tp = %(prob).3f\n" % comp)



if __name__ == '__main__':
	@cubify.Cubify
	def twopeakIntegrand(x):
		a = 1e-5
		pref = 1 / (a * np.pi) 
		peaklist = np.sort([0.12312, 1.12413, 0.13123, 0.43535, 0.657575,0.12312,0.34535,0.25363,0.536344,0.235235,0.24353,0.78878,0.89898435,0.3423,0.567,0.4333])
		# result = pref * 1 / ( 1 + ((x-0.1312)**2 / a**2)) + pref * 1 / ( 1 + ((x-0.73217)**2 / a**2))
		result = np.sum([pref * 1 / ( 1 + ((x-x0)**2 / a**2)) for x0 in peaklist]) 
		return result

	@cubify.Cubify
	def CMPLXpeakIntegrand(x):
		k = 2
		a = 1e-5
		pref = 1 / (a * np.pi) 
		peaklist = np.sort([0.12312, 1.12413, 0.13123, 0.43535, 0.657575,0.12312,0.34535,0.25363,0.536344,0.235235,0.24353,0.78878,0.89898435,0.3423,0.567,0.4333])
		# result = pref * 1 / ( 1 + ((x-0.1312)**2 / a**2)) + pref * 1 / ( 1 + ((x-0.73217)**2 / a**2))
		result = np.sum([pref * 1 / ( 1 + ((x-x0)**2 / a**2)) for x0 in peaklist]) 
		return result * np.exp(-1j * k * x)
	@cubify.Cubify
	def REALpeakIntegrand(x):
		k = 2
		a = 1e-5
		pref = 1 / (a * np.pi) 
		peaklist = np.sort([0.12312, 1.12413, 0.13123, 0.43535, 0.657575,0.12312,0.34535,0.25363,0.536344,0.235235,0.24353,0.78878,0.89898435,0.3423,0.567,0.4333])
		# result = pref * 1 / ( 1 + ((x-0.1312)**2 / a**2)) + pref * 1 / ( 1 + ((x-0.73217)**2 / a**2))
		result = np.sum([pref * 1 / ( 1 + ((x-x0)**2 / a**2)) for x0 in peaklist]) 
		return result * np.cos(k * x)
	@cubify.Cubify
	def IMAGpeakIntegrand(x):
		k = 2
		a = 1e-5
		pref = 1 / (a * np.pi) 
		peaklist = np.sort([0.12312, 1.12413, 0.13123, 0.43535, 0.657575,0.12312,0.34535,0.25363,0.536344,0.235235,0.24353,0.78878,0.89898435,0.3423,0.567,0.4333])
		# result = pref * 1 / ( 1 + ((x-0.1312)**2 / a**2)) + pref * 1 / ( 1 + ((x-0.73217)**2 / a**2))
		result = np.sum([pref * 1 / ( 1 + ((x-x0)**2 / a**2)) for x0 in peaklist]) 
		return result * -1. * np.sin(k * x)
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
	# Integrand = twopeakIntegrand
	Integrand = CMPLXpeakIntegrand
	KEY = 0
	verbose=0

	# print_header('Vegas')
	# print_results('Vegas', pycuba.Vegas(Integrand, NDIM, verbose=verbose, maxeval=MAXEVAL,  epsrel=1e-3))
# 
	# print_header('Suave')
	# print_results('Suave', pycuba.Suave(Integrand, NDIM, NNEW, NMIN, FLATNESS, verbose=verbose,maxeval = MAXEVAL,  epsrel=1e-3))

	# print_header('Divonne')
	# print_results('Divonne', pycuba.Divonne(Integrand, NDIM, 
				# mineval=MINEVAL, maxeval=MAXEVAL,
				# key1=KEY1, key2=KEY2, key3=KEY3, maxpass=MAXPASS,
				# border=BORDER, maxchisq=MAXCHISQ, mindeviation=MINDEVIATION,
				# ldxgiven=LDXGIVEN, verbose=verbose,epsrel=1e-3))

	print_header('CuhreCMPLX')
	print_results('Cuhre', pycuba.Cuhre(CMPLXpeakIntegrand, NDIM, key=KEY, maxeval=MAXEVAL, verbose=verbose,epsrel=1e-6 ))
	print_header('CuhreREAL')
	print_results('Cuhre', pycuba.Cuhre(REALpeakIntegrand, NDIM, key=KEY, maxeval=MAXEVAL, verbose=verbose,epsrel=1e-6 ))
	print_header('CuhreIMAG')
	print_results('Cuhre', pycuba.Cuhre(IMAGpeakIntegrand, NDIM, key=KEY, maxeval=MAXEVAL, verbose=verbose,epsrel=1e-6 ))

