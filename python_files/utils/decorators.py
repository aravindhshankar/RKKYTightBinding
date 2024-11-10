from functools import wraps
import pycuba
import numpy as np
import math

class cubify:
	# Class attributes for constants
	LOWERLIM = 0
	UPPERLIM = 1

	@classmethod
	def set_limits(cls, a_, b_):
		cls.LOWERLIM = a_
		cls.UPPERLIM = b_

	# Decorator that uses the constants
	@classmethod
	def Cubify(cls, f):
		@wraps(f)
		def wrapper(ndim,xx,ncomp,ff,userdata):
			x,_ = [xx[i] for i in range(ndim.contents.value)]
			ff[0] = (cls.UPPERLIM-cls.LOWERLIM) * f(cls.LOWERLIM + (cls.UPPERLIM-cls.LOWERLIM)*x)
			return 0
		return wrapper 

