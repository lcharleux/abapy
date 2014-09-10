# Importing packages
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import leastsq

# Setting up the database
execfile('settings.py')



# creating some shortcuts
d = db_manager     # database manager
c = db_manager.cls # useful to call Simulation attributs 
# Getting simulations:
simus = d.query().filter(c.completed == True).all()

unload = 2
plt.figure(0)
plt.clf()
S = []

for simu in simus: 
  disp = np.array(simu.disp_hist[unload].data[0])
  force = np.array(simu.force_hist[unload].data[0])
  max_force = force.max()
  max_disp = disp.max()
  loc = np.where(force >= max_force * .1)
  disp = disp[loc] /max_disp
  force = force[loc] / max_force
  func = lambda k, x: ( (x - k[0]) / (1. - k[0] ) )**k[1] 
  err = lambda v, x, y: (func(v,x)-y)
  k0 = [0., 1.]
  k, success = leastsq(err, k0, args=(disp,force), maxfev=10000)
  force_fit = func(k,disp)
  plt.plot(disp, force, 'or')
  plt.plot(disp, force_fit, 'b-')
  S.append(k[1] / (1. - k[0]) )
  

plt.show() 
