from abapy.mesh import RegularQuadMesh
from abapy.indentation import IndentationMesh
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.collections as collections
import numpy as np
from matplotlib import cm
from scipy import interpolate

def vector_function(x, y, z, labels):
  """
  Vector function used to produced the displacement field.
  """
  r0 = 1.
  theta = .5 * np.pi * x
  r = y + r0
  ux = -x + r * np.cos(theta**2)
  uy = -y + r * np.sin(theta**2)
  uz = 0. * z
  return ux, uy, uz

def scalar_function(x, y, z, labels):
  """
  Scalar function used to produced the plotted field.
  """
  return x**2 + y**2
#MESH GENERATION
N1, N2 = 30, 30
l1, l2 = .75, 1.
m = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)
#FIELDS GENERATION
u = m.nodes.eval_vectorFunction(vector_function)
m.add_field(u, "u")
f = m.nodes.eval_function(scalar_function)
m.add_field(f, "f")
#PLOTS
fig = plt.figure(0)
plt.clf()
ax = fig.add_subplot(1,1,1)
m.draw(ax, 
    disp_func  = lambda fields : fields["u"],
    field_func = lambda fields : fields["f"],
    cmap = cm.jet,
    cbar_orientation = "vertical",
    contour = False,
    contour_colors = "black",
    alpha = 1.,
    cmap_levels = 10,
    edge_width = .1)
ax.set_aspect("equal")
plt.grid()
plt.xlabel("$x$ position")
plt.ylabel("$y$ position")
plt.show()

