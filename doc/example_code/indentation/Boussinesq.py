from abapy.mesh import Mesh, Nodes, RegularQuadMesh
from abapy import CFE
from numpy import linspace
import matplotlib.pyplot as plt

"""
Bousinesq model
===============
"""
Bo = CFE.Boussinesq(F = 10.)

"""
Mesh model
"""
N1, N2 = 50, 50
l1, l2 = 1., 1.
Ncolor = 200
levels = linspace(0., 10., 20)

mesh = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)
nodes = mesh.nodes

tensorField = mesh.nodes.eval_tensorFunction(Bo.sigma_boussinesq())

sigma_zz = tensorField.get_component(22) 
VM = tensorField.vonmises() 
P = tensorField.pressure() 

xt,yt,zt = mesh.convert2tri3().get_edges() 
xb,yb,zb = mesh.get_border()
X,Y,Z,tri = mesh.dump2triplot()

"""
Display of stress fields
"""
fig = plt.figure(figsize=(16,4))

g1 = fig.add_subplot(131)
g1.set_aspect('equal')
g1.set_title(r'$\sigma_{zz}$')
g1.tricontourf(X,Y,tri,sigma_zz.data, levels = levels)
g1.tricontour(X,Y,tri,sigma_zz.data, levels = levels, colors = 'black')

g2 = fig.add_subplot(132)
g2.set_aspect('equal')
g2.set_title(r'Von Mises $\sigma_{eq}$')
g2.tricontourf(X,Y,tri,VM.data, levels = levels)
g2.tricontour(X,Y,tri,VM.data, levels = levels, colors = 'black')

g3 = fig.add_subplot(133)
g3.set_aspect('equal')
g3.set_title(r'Pressure $p$')
g3.tricontourf(X,Y,tri,P.data, levels = sorted(-levels))
g3.tricontour(X,Y,tri,P.data, levels = sorted(-levels), colors = 'black')

plt.show()
