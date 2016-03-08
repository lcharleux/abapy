from abapy.mesh import RegularQuadMesh
N1, N2 = 1,1
mesh = RegularQuadMesh(N1, N2)
mesh.replace_node(1,2)
print mesh 
