from abapy.misc import load
import numpy as np
from matplotlib import pyplot as plt


# In this case, a 3D FEM simulation has beed performed and the results are stored in the file ``ContactData_berk.pckl``. See ``Get_ContactData`` to understand how this data has been extracted from an Abaqus odb file.


out = load('ContactData_berk.pckl')
cd0 = out[1][-1] # First step data: loading
cd1 = out[2][-1] # Second step data: unloading
hmax = -cd0.min_altitude()

p2, p1, p0 = cd0.contact_contour()
x0, y0 = p0[:,0], p0[:,1]
x1, y1 = p1[:,0], p1[:,1]
x2, y2 = p2[:,0], p2[:,1]


plt.figure()
plt.clf()
plt.title('Contact area contour')
plt.plot(x0, y0, label = 'upper bound')
plt.plot(x1, y1, label = 'middle')
plt.plot(x2, y2, label = 'lower bound')
plt.grid()
plt.legend()
plt.show()
