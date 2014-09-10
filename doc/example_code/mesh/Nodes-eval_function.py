from abapy.mesh import Mesh, Nodes, RegularQuadMesh
import matplotlib.pyplot as plt 
from numpy import cos, pi
def function(x, y, z, labels):
 r = (x**2 + y**2)**.5
 return cos(2*pi*x)*cos(2*pi*y)/(r+1.)
N1, N2 = 100, 25
l1, l2 = 4., 1.
Ncolor = 20
mesh = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)
field = mesh.nodes.eval_function(function)
x,y,z = mesh.get_edges() # Mesh edges
X,Y,Z,tri = mesh.dump2triplot()
xb,yb,zb = mesh.get_border()
fig = plt.figure(figsize=(16,4))
fig.gca().set_aspect('equal')
fig.frameon = True
plt.plot(xb,yb,'k-', linewidth = 2.)
plt.xticks([0,l1],['$0$', '$l_1$'], fontsize = 15.)
plt.yticks([0,l2],['$0$', '$l_2$'], fontsize = 15.)
plt.tricontourf(X,Y,tri,field.data, Ncolor)
plt.tricontour(X,Y,tri,field.data, Ncolor, colors = 'black')
plt.show()

