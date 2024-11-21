from functools import wraps

class cubify:
    # Class attributes for constans indicating range of integration
    LOWERLIM = 0.
    UPPERLIM = 1.

    @classmethod
    def set_limits(cls, a_, b_):
        cls.LOWERLIM = a_
        cls.UPPERLIM = b_

    # Decorator that returns a function of one variable, scales it appropriately for the cuba integration routines
    @classmethod
    def Cubify(cls, f):
        @wraps(f)
        def wrapper(ndim,xx,ncomp,ff,userdata):
            x,_ = [xx[i] for i in range(ndim.contents.value)]
            ff[0] = (cls.UPPERLIM-cls.LOWERLIM) * f(cls.LOWERLIM + (cls.UPPERLIM-cls.LOWERLIM)*x)
            return 0
        return wrapper 

    # When the function returns a vector 
    @classmethod
    def VECCubify(cls, f):
        @wraps(f)
        def wrapper(ndim,xx,ncomp,ff,userdata):
            x,_ = [xx[i] for i in range(ndim.contents.value)]
            # ff = (cls.UPPERLIM-cls.LOWERLIM) * f(cls.LOWERLIM + (cls.UPPERLIM-cls.LOWERLIM)*x)
            ffO = (cls.UPPERLIM-cls.LOWERLIM) * f(cls.LOWERLIM + (cls.UPPERLIM-cls.LOWERLIM)*x)
            for j in range(ncomp.contents.value):
                ff[j] = ffO[j]
            return 0
        return wrapper 
