import numpy as np 

def f(x,y): #function to be parallelized on xlist and ylist 
    return x + y 



tlist = np.array([0.021,0.022,0.024,0.026,0.028,0.029,0.030])
betalist = 1./tlist
print(betalist)
