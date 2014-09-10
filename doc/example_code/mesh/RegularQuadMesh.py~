from abapy.mesh import RegularQuadMesh
from matplotlib import pyplot as plt
N1,N2 = 30,5 # Number of elements
l1, l2 = 4., 1. # Mesh size
fs = 20. # fontsize
mesh = RegularQuadMesh(N1,N2,l1,l2)
plt.figure(figsize=(8,3))
plt.gca().set_aspect('equal')
nodes = mesh.nodes
xn, yn, zn = nodes.x, nodes.y, nodes.z # Nodes coordinates
xe,ye,ze = mesh.get_edges() # Mesh edges
xb,yb,zb = mesh.get_border() # Mesh border
plt.plot(xe,ye,'r-',label = 'edges')
plt.plot(xb,yb,'b-',label = 'border')
plt.plot(xn,yn,'go',label = 'nodes')
plt.xlim([-.1*l1,1.1*l1])
plt.ylim([-.1*l2,1.1*l2])
plt.xticks([0,l1],['$0$', '$l_1$'],fontsize = fs)
plt.yticks([0,l2],['$0$', '$l_2$'],fontsize = fs)
plt.xlabel('$N_1$',fontsize = fs)
plt.ylabel('$N_2$',fontsize = fs)
plt.legend()
plt.show()

