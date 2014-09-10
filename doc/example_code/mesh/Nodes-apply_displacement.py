from abapy.mesh import Mesh, Nodes, RegularQuadMesh
import matplotlib.pyplot as plt 
from numpy import cos, sin, pi
from copy import copy
def function(x, y, z, labels):
  r0 = 1.
  theta = 2 * pi * x
  r = y + r0
  ux = -x + r * cos(theta)
  uy = -y + r * sin(theta)
  uz = 0. * z
  return ux, uy, uz
N1, N2 = 100, 25
l1, l2 = .75, 1.
Ncolor = 20
mesh = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)
vectorField = mesh.nodes.eval_vectorFunction(function)
mesh.nodes.apply_displacement(vectorField)
field = vectorField.get_coord(2) # we chose to plot coordinate 2
field2 = vectorField.get_coord(2) # we chose to plot coordinate 2
x,y,z = mesh.get_edges() # Mesh edges
X,Y,Z,tri = mesh.dump2triplot()
xb,yb,zb = mesh.get_border() # mesh borders
xe, ye, ze = mesh.get_edges()
fig = plt.figure(figsize=(10,10))
fig.gca().set_aspect('equal')
plt.axis('off')
plt.plot(xb,yb,'k-', linewidth = 2.)
plt.plot(xe, ye,'k-', linewidth = .5)
plt.tricontour(X,Y,tri,field.data, Ncolor, colors = 'black')
color = plt.tricontourf(X,Y,tri,field.data, Ncolor)
plt.colorbar(color)
plt.show()

