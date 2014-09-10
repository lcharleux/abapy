from abapy.mesh import RegularQuadMesh
from matplotlib import pyplot as plt
N1,N2 = 20,20
l1, l2 = 1., 1.
mesh1 = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2, name = 'mesh1_el')
mesh2 = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2, name =  'mesh2_el')
mesh2.add_set('set2',[1,3])

mesh2.nodes.translate(x = l1, y = l2)

mesh1.union(mesh2)



plt.figure()
xe, ye, ze = mesh1.get_edges()
plt.plot(xe, ye)
plt.show()
