from abapy.indentation import DeformableCone3D
from matplotlib import pyplot as plt
c = DeformableCone3D(Na =4, Nb = 4, Ns = 1, N = 10, sweep_angle = 120.)
c.make_mesh()
f = open('DeformableCone3D.vtk', 'w')
f.write(c.mesh.dump2vtk())
f.close()
x,y,z = c.mesh.get_edges()
# Adding some 3D "home made" perspective:
zx, zy = .3, .3
for i in xrange(len(x)):
  if x[i] != None:
    x[i] += zx * z[i]
    y[i] += zy * z[i]
plt.plot(x,y)
plt.show()
