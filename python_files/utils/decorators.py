from functools import wraps
import pycuba
import numpy as np
import math

class cubadec:
    # Class attributes for constants
    LOWERLIM = 0
    UPPERLIM = 1

    @classmethod
    def set_constants(cls, a_, b_):
        cls.LOWERLIM = a_
        cls.UPPERLIM = b_

    # Decorator one that uses the constants
    @classmethod
    def cubify(cls, f):
        @wraps(f)
		def wrapper(ndim,xx,ncomp,ff,userdata):
			x,_ = [xx[i] for i in range(ndim.contents.value)]
			ff[0] = (UPPERLIM-LOWERLIM) * f(LOWERLIM + (UPPERLIM-LOWERLIM)*x)
			return 0
		return wrapper 

