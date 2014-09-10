from abapy.mesh import RegularQuadMesh
N = 1
l1, l2 = 1., 2.
mesh = RegularQuadMesh(N1 = N, N2 = N, l1 = l1, l2 = l2)
mesh.nodes.closest_node(1)
