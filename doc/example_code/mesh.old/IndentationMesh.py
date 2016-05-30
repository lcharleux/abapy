from abapy.mesh import IndentationMesh
from matplotlib import pyplot as plt
Na = 2
Nb = 3

l, ltot = 1., 2
mesh = IndentationMesh(Na,Nb,l)
'''
xe, ye, ze = mesh.get_edges()
plt.figure()
plt.gca().set_aspect('equal')
plt.plot(xe, ye, 'r-')
plt.plot(mesh.nodes.x, mesh.nodes.y, 'ob')
bbox = mesh.nodes.boundingBox()
plt.xlim(bbox[0])
plt.ylim(bbox[1])
plt.show()
'''
