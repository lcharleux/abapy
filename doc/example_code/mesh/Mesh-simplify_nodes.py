from abapy.mesh import RegularQuadMesh
N1,N2 = 2,2
l1, l2 = 1., 1.
mesh1 = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)
mesh2 = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2)

mesh2.translate(x = l1, y = l2)
