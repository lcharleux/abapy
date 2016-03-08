from abapy.mesh import RegularQuadMesh
from abapy.indentation import IndentationMesh
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.collections as collections
import mpl_toolkits.mplot3d as a3
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
N1, N2, N3 = 10, 10, 5
l1, l2, l3 = .75, 1., 1.



m = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)
m = m.extrude(l = l3, N = N3 )
vectorField = m.nodes.eval_vectorFunction(function)
m.nodes.apply_displacement(vectorField)
patches = m.dump2polygons(use_3D = True, 
                          face_color = None, 
                          edge_color = "black")
bb = m.nodes.boundingBox()
patches.set_linewidth(1.)

fig = plt.figure(0)
plt.clf()
ax = a3.Axes3D(fig)
ax.set_aspect("equal")
ax.add_collection3d(patches)
plt.xlim(bb[0])
plt.ylim(bb[1])
plt.xlabel("$x$ position")
plt.ylabel("$y$ position")
plt.show()

