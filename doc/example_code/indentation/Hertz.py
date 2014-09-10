from abapy.indentation import Hertz
from abapy.mesh import Mesh, Nodes, RegularQuadMesh
from matplotlib import pyplot as plt 
import numpy as np

"""
===========
Hertz model
===========
"""
    
H = Hertz(F = 1., E=1., nu = 0.1)
Ne = 50

mesh = RegularQuadMesh(N1 = Ne, N2 = Ne, l1 = H.a * 2., l2 = H.a * 2., dtf = 'd') 
mesh.nodes.translate(H.a/20., H.a/20.)
S = mesh.nodes.eval_tensorFunction(H.sigma)
R,Z,T,tri = mesh.dump2triplot()
R, Z = np.array(R), np.array(Z)
# Some fields
srr = S.get_component(11)
szz = S.get_component(22)
stt = S.get_component(33)
srz = S.get_component(12)
smises = S.vonmises()
s1, s2, s3, v1, v2, v3 = S.eigen() # Eigenvalues and eigenvectors
data = smises.data

N = 20
levels = np.linspace(0., max(data), N)
a = H.a
plt.figure()
plt.tricontourf(R/a, Z/a, tri, data, levels)
plt.colorbar()
plt.tricontour(R/a, Z/a, tri, data, levels, colors = 'black')
plt.xlabel('$r/a$', fontsize = 14.)
plt.ylabel('$z/a$', fontsize = 14.)
#plt.quiver(R, Z, v1.data1, v1.data2)
plt.grid()
plt.show()

