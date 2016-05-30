from abapy.mesh import RegularQuadMesh
from abapy.indentation import IndentationMesh
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.collections as collections
import numpy as np
from matplotlib import cm
from scipy import interpolate


def function(x, y, z, labels):
  r0 = 1.
  theta = .5 * np.pi * x
  r = y + r0
  ux = -x + r * np.cos(theta**2)
  uy = -y + r * np.sin(theta**2)
  uz = 0. * z
  return ux, uy, uz
N1, N2 = 30, 30
l1, l2 = .75, 1.



m = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)
vectorField = m.nodes.eval_vectorFunction(function)
m.nodes.apply_displacement(vectorField)
patches = m.dump2polygons()
volume = m.volume()
bb = m.nodes.boundingBox()
patches.set_facecolor(None) # Required to allow face color
patches.set_array(volume)
patches.set_linewidth(1.)
fig = plt.figure(0)
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect("equal")
ax.add_collection(patches)
plt.legend()
cbar = plt.colorbar(patches)
plt.grid()
plt.xlim(bb[0])
plt.ylim(bb[1])
plt.xlabel("$x$ position")
plt.ylabel("$y$ position")
cbar.set_label("Element volume")
plt.show()

