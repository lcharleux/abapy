from abapy.mesh import RegularQuadMesh
from matplotlib import pyplot as plt
# Creating a mesh
m = RegularQuadMesh(N1 = 2, N2 = 2)
x0, y0, z0 = m.get_edges()
# Finding the node located at x = y =0.:
nodes = m.nodes
for i in xrange(len(nodes.labels)):
  if nodes.x[i] == 0. and nodes.y[i] == 0.: node = nodes.labels[i]
# Removing this node
m.drop_node(node)
x1, y1, z1 = m.get_edges()
bbx, bby, bbz = m.nodes.boundingBox()
plt.figure()
plt.clf()
plt.gca().set_aspect('equal')
plt.axis('off')
plt.xlim(bbx)
plt.ylim(bby)
plt.plot(x0,y0, 'r-', linewidth = 2., label = 'Removed element')
plt.plot(x1,y1, 'b-', linewidth = 2., label = 'New mesh')
plt.legend()
plt.show()


