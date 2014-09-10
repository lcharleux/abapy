# PYTHON POST PROCESSING SCRIPT
# Run using python

# Packages
from abapy.misc import load
import matplotlib.pyplot as plt
import numpy as np

# Setting up some pathes
workdir = 'workdir'
name = 'indentation_axi'

# Getting back raw data
data = load(workdir + '/' + name + '.pckl')

# Post processing
ref_node_label = data['ref_node_label']
force_hist = -data['RF2']['Node I_INDENTER.{0}'.format(ref_node_label)]
disp_hist = -data['U2']['Node I_INDENTER.{0}'.format(ref_node_label)]

time, force_l = force_hist[0,1].plotable() # Getting back force  during loading
time, disp_l = disp_hist[0,1].plotable()   # Getting backdisplacement during loading
time, force_u = force_hist[2].plotable() # Getting back force  during unloading
time, disp_u = disp_hist[2].plotable()   # Getting backdisplacement during unloading

# Dimensional analysis (E. Buckingham. Physical review, 4, 1914.) shows that the loading curve must be parabolic, let's build a parabolic fit 


C_factor = (force_hist[1] / disp_hist[1]**2).average()
disp_fit = np.array(disp_hist[0,1].toArray()[1])
force_fit = C_factor * disp_fit**2


plt.figure()
plt.clf()
plt.plot(disp_l, force_l, 'r-', linewidth = 2., label = 'Loading')
plt.plot(disp_u, force_u, 'b-', linewidth = 2., label = 'Unloading')
plt.plot(disp_fit, force_fit, 'g-', linewidth = 2., label = 'Parabolic fit')
plt.xlabel('Displacement $d$')
plt.ylabel('Force $F$')
plt.grid()
plt.legend(loc = 'upper left')
plt.show()
