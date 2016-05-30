from abapy.mesh import Mesh
from matplotlib import pyplot as plt
import numpy as np

N1,N2 = 10,5 # Number of elements
l1, l2 = 4., 2. # Mesh size
fs = 20. # fontsize
mesh = Mesh()
nodes =  mesh.nodes
nodes.add_node(label = 1, x = 0., y = 0.)
nodes.add_node(label = 2, x = 1., y = 0.)
nodes.add_node(label = 3, x = 0., y = 1.)
nodes.add_node(label = 4, x = 1.5, y = 1.)
nodes.add_node(label = 5, x = 1., y = 2.)
mesh.add_element(label = 1, connectivity = [1,2,3], space = 2)
mesh.add_element(label = 2, connectivity = [2,4,5,3], space = 2)


centroids = mesh.centroids()
     
plt.figure(figsize=(8,3))
plt.gca().set_aspect('equal')
nodes = mesh.nodes
xn, yn, zn = np.array(nodes.x), np.array(nodes.y), np.array(nodes.z) # Nodes coordinates
xe,ye,ze = mesh.get_edges() # Mesh edges
xb,yb,zb = mesh.get_border() # Mesh border


plt.plot(xe,ye,'r-',label = 'Edges')
plt.plot(xb,yb,'b-',label = 'Border')
plt.plot(xn,yn,'go',label = 'Nodes')
plt.xlim([-.1*l1,1.1*l1])
plt.ylim([-.1*l2,1.1*l2])
plt.xlabel('$x$',fontsize = fs)
plt.ylabel('$y$',fontsize = fs)
plt.plot(centroids[:,0], centroids[:,1], '*', label = "Centroids")
plt.legend()
plt.grid()
plt.show()
