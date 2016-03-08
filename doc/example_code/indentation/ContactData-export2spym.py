from abapy.misc import load
import numpy as np
import matplotlib.pyplot as plt

# In this case, a 3D FEM simulation has beed performed and the results are stored in the file ``ContactData.pckl``. See ``Get_ContactData`` to understand how this data has been extracted from an Abaqus odb file.


out = load('ContactData_berk.pckl')
cdl = out[1][-1] # loading
cdu = out[2][-1] # unloading
hmax = -cdl.min_altitude()
l = 7. * hmax
alt, press = cdu.export2spym(lx = l, ly = l, nx = 512, ny = 512)
alt.dump2gsf('image.gsf')
xlabel, ylabel, zlabel = alt.get_labels()
X,Y,Z = alt.get_xyz()
plt.figure()
plt.clf()
plt.gca().set_aspect('equal')
plt.grid()
plt.contourf(X, Y, Z, 10)
cbar = plt.colorbar()
cbar.set_label(zlabel)
plt.contour(X, Y, Z, 10, colors = 'black')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()


