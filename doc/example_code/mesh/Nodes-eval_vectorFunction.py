from abapy.mesh import Mesh, Nodes, RegularQuadMesh
import matplotlib.pyplot as plt 
from numpy import cos, sin, pi
def function(x, y, z, labels):
  r0 = 1.
  theta = 2 * pi * x
  r = y + r0
  ux = -x + r * cos(theta)
  uy = -y + r * sin(theta)
  uz = 0. * z
  return ux, uy, uz
N1, N2 = 100, 25
l1, l2 = 1., 1.
Ncolor = 20
mesh = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)
vectorField = mesh.nodes.eval_vectorFunction(function)
field = vectorField.get_coord(1) # we chose to plot coordinate 1
field2 = vectorField.get_coord(2) # we chose to plot coordinate 1
field3 = vectorField.norm() # we chose to plot norm
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax.set_frame_on(False)
ax2.set_frame_on(False)
ax3.set_frame_on(False)
ax.set_title(r'$V_1$')
ax2.set_title(r'$V_2$')
ax3.set_title(r'$\sqrt{\vec{V}^2}$')
ax3.set_title(r'$||\vec{V}||$')
x,y,z = mesh.get_edges() # Mesh edges
xt,yt,zt = mesh.convert2tri3().get_edges() # Triangular mesh edges
xb,yb,zb = mesh.get_border()
X,Y,Z,tri = mesh.dump2triplot()
ax.plot(xb,yb,'k-', linewidth = 2.)
ax.tricontourf(X,Y,tri,field.data, Ncolor)
ax.tricontour(X,Y,tri,field.data, Ncolor, colors = 'black')
ax2.plot(xb,yb,'k-', linewidth = 2.)
ax2.tricontourf(X,Y,tri,field2.data, Ncolor)
ax2.tricontour(X,Y,tri,field2.data, Ncolor, colors = 'black')
ax3.plot(xb,yb,'k-', linewidth = 2.)
ax3.tricontourf(X,Y,tri,field3.data, Ncolor)
ax3.tricontour(X,Y,tri,field3.data, Ncolor, colors = 'black')
ax.set_xlim([-.1*l1,1.1*l1])
ax.set_ylim([-.1*l2,1.1*l2])
ax2.set_xlim([-.1*l1,1.1*l1])
ax2.set_ylim([-.1*l2,1.1*l2])
ax3.set_xlim([-.1*l1,1.1*l1])
ax3.set_ylim([-.1*l2,1.1*l2])
plt.show()
