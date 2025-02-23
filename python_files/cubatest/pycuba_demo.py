from __future__ import absolute_import, unicode_literals, print_function
import os
# virtualenv_lib_path = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/.BLGvenv/lib/cubalib'
# os.environ['DYLD_LIBRARY_PATH'] = f"{virtualenv_lib_path}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"  # macOS
import pycuba
#pycuba.demo()

if __name__ == '__main__':
  import math

  def Integrand(ndim, xx, ncomp, ff, userdata):
    x,y,z = [xx[i] for i in range(ndim.contents.value)]
    result = math.sin(x)*math.cos(y)*math.exp(z)
    ff[0] = result
    return 0
    

  NDIM = 3
  NCOMP = 1

  NNEW = 1000
  NMIN = 2
  FLATNESS = 50.

  KEY1 = 47
  KEY2 = 1
  KEY3 = 1
  MAXPASS = 5
  BORDER = 0.
  MAXCHISQ = 10.
  MINDEVIATION = .25
  NGIVEN = 0
  LDXGIVEN = NDIM
  NEXTRA = 0
  MINEVAL = 0
  MAXEVAL = 50000


  KEY = 0
  
  def print_header(name):
    print('-------------------- %s test -------------------' % name)
  def print_results(name, results):
    keys = ['nregions', 'neval', 'fail']
    text = ["%s %d" % (k, results[k]) for k in keys if k in results]
    print("%s RESULT:\t" % name.upper() + "\t".join(text))
    for comp in results['results']:
      print("%s RESULT:\t" % name.upper() + \
	"%(integral).8f +- %(error).8f\tp = %(prob).3f\n" % comp)
  
  from os import environ as env
  verbose = 2
  if 'CUBAVERBOSE' in env:
    verbose = int(env['CUBAVERBOSE'])
  
  print_header('Vegas')
  print_results('Vegas', pycuba.Vegas(Integrand, NDIM, verbose=2))
  
  print_header('Suave')
  print_results('Suave', pycuba.Suave(Integrand, NDIM, NNEW, NMIN, FLATNESS, verbose=2 | 4))
  
  print_header('Divonne')
  print_results('Divonne', pycuba.Divonne(Integrand, NDIM, 
    mineval=MINEVAL, maxeval=MAXEVAL,
    key1=KEY1, key2=KEY2, key3=KEY3, maxpass=MAXPASS,
    border=BORDER, maxchisq=MAXCHISQ, mindeviation=MINDEVIATION,
    ldxgiven=LDXGIVEN, verbose=2))
  
  print_header('Cuhre')
  print_results('Cuhre', pycuba.Cuhre(Integrand, NDIM, key=KEY, verbose=2 | 4))

