from abapy.mesh import Mesh, Nodes, RegularQuadMesh
import matplotlib.pyplot as plt 
from numpy import cos, sin, pi, linspace
def boussinesq(r, z, theta, labels):
  '''
  Stress solution of the Boussinesq point loading of semi infinite elastic half space for a force F = 1. and nu = 0.3
  '''
  from math import pi
  from numpy import zeros_like
  nu = 0.3
  rho = (r**2 + z**2)**.5
  s_rr = -1./(2. * pi * rho**2) * ( (-3. * r**2 * z)/(rho**3) + (1.-2. * nu)*rho / (rho + z) )
  #s_rr = 1./(2.*pi) *( (1-2*nu) * ( r**-2 -z / (rho * r**2)) - 3 * z * r**2 / rho**5 )
  s_zz = 3. / (2. *pi ) * z**3 / rho**5
  s_tt = -( 1. - 2. * nu) / (2. * pi * rho**2 ) * ( z/rho - rho / (rho + z) )
  #s_tt = ( 1. - 2. * nu) / (2. * pi ) * ( 1. / r**2 -z/( rho * r**2) -z / rho**3  )
  s_rz = -3./ (2. * pi) * r * z**2 / rho **5
  s_rt = zeros_like(r)
  s_zt = zeros_like(r)
  return s_rr, s_zz, s_tt, s_rz, s_rt, s_zt
 
  return ux, uy, uz


N1, N2 = 50, 50
l1, l2 = 1., 1.
Ncolor = 200
levels = linspace(0., 10., 20)

mesh = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)
# Finding the node located at x = y =0.:
nodes = mesh.nodes
for i in xrange(len(nodes.labels)):
  if nodes.x[i] == 0. and nodes.y[i] == 0.: node = nodes.labels[i]
mesh.drop_node(node)
tensorField = mesh.nodes.eval_tensorFunction(boussinesq)
field = tensorField.get_component(22) # sigma_zz
field2 = tensorField.vonmises() # von Mises stress
field3 = tensorField.pressure() # pressure

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
ax.set_title(r'$\sigma_{zz}$')
ax2.set_title(r'Von Mises $\sigma_{eq}$')
ax3.set_title(r'Pressure $p$')
xt,yt,zt = mesh.convert2tri3().get_edges() # Triangular mesh edges
xb,yb,zb = mesh.get_border()

X,Y,Z,tri = mesh.dump2triplot()

ax.plot(xb,yb,'k-', linewidth = 2.)
ax.tricontourf(X,Y,tri,field.data, levels = levels)
ax.tricontour(X,Y,tri,field.data, levels = levels, colors = 'black')
ax2.plot(xb,yb,'k-', linewidth = 2.)
ax2.tricontourf(X,Y,tri,field2.data, levels = levels)
ax2.tricontour(X,Y,tri,field2.data, levels = levels, colors = 'black')
ax3.plot(xb,yb,'k-', linewidth = 2.)
ax3.tricontourf(X,Y,tri,field3.data, levels = sorted(-levels))
ax3.tricontour(X,Y,tri,field3.data, levels = sorted(-levels), colors = 'black')
ax.set_xlim([-.1*l1,1.1*l1])
ax.set_ylim([-.1*l2,1.1*l2])
ax2.set_xlim([-.1*l1,1.1*l1])
ax2.set_ylim([-.1*l2,1.1*l2])
ax3.set_xlim([-.1*l1,1.1*l1])
ax3.set_ylim([-.1*l2,1.1*l2])
plt.show()

