from abapy.misc import load
import numpy as np
from matplotlib import pyplot as plt


# In this case, a 3D FEM simulation has beed performed and the results are stored in the file ``ContactData_berk.pckl``. See ``Get_ContactData`` to understand how this data has been extracted from an Abaqus odb file.


out = load('ContactData_berk.pckl')
cd0 = out[1][-1] # First step data: loading
cd1 = out[2][-1] # Second step data: unloading
hmax = -cd0.min_altitude()

# First let's map altitude and pressure on cartesian grids.
x = np.linspace(-2., 2., 256)
X, Y = np.meshgrid(x, x)

Alt0, Press0 = cd0.interpolate(X, Y, method ='linear')
Alt1, Press1 = cd1.interpolate(X, Y, method ='linear')
Alt0 = Alt0 / hmax
Alt1 = Alt1 / hmax

# Now we wan to get some sections of the imprint
s = np.linspace(0., 2., 256)
s = np.append((-s)[::-1], s)
theta0 = np.radians(0.01)
theta1 = np.radians(15.)
xs0 = np.cos(theta0) * s 
ys0 = np.sin(theta0) * s
xs1 = np.cos(theta1) * s 
ys1 = np.sin(theta1) * s
# Sections under full load
Alt0_sec_l, Press0_sec_l = cd0.interpolate(xs0, ys0, method ='linear')
Alt1_sec_l, Press1_sec_l = cd0.interpolate(xs1, ys1, method ='linear')
Alt0_sec_l = Alt0_sec_l / hmax
Alt1_sec_l = Alt1_sec_l / hmax
# Sections after unloading
Alt0_sec_u, Press0_sec_u = cd1.interpolate(xs0, ys0, method ='linear')
Alt1_sec_u, Press1_sec_u = cd1.interpolate(xs1, ys1, method ='linear')
Alt0_sec_u = Alt0_sec_u / hmax
Alt1_sec_u = Alt1_sec_u / hmax





fig = plt.figure()
plt.clf()
ax1 = fig.add_subplot(221)
ax1.set_xticks([])
ax1.set_yticks([])
grad = plt.contourf(X, Y, Alt0, 10)
plt.contour(X, Y, Alt0, 10, colors = 'black')
plt.grid()
plt.plot(xs0, ys0, 'b-')
plt.plot(xs1, ys1, 'r-')
ax1.set_aspect('equal')
plt.title('Altitude Loaded')
ax2 = fig.add_subplot(222)
ax2.set_xticks([])
ax2.set_yticks([])
grad = plt.contourf(X, Y, Alt1, 10)
plt.contour(X, Y, Alt1, 10, colors = 'black')
plt.grid()
plt.plot(xs0, ys0, 'b-')
plt.plot(xs1, ys1, 'r-')
ax2.set_aspect('equal')
plt.title('Altitude Unloaded')
ax3 = fig.add_subplot(223)
ax3.set_ylim([-1,0.3])
plt.plot(s, Alt0_sec_l, 'b-')
plt.plot(s, Alt1_sec_l, 'r-')
plt.title('Cross sections')
plt.grid()
ax4 = fig.add_subplot(224)
ax4.set_ylim([-1,0.3])
plt.plot(s, Alt0_sec_u, 'b-')
plt.plot(s, Alt1_sec_u, 'r-')
plt.title('Cross sections')
plt.grid()
plt.show()


