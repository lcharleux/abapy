from abapy.mesh import RegularQuadMesh
N1, N2 = 1,1
mesh = RegularQuadMesh(N1, N2)
nodes = mesh.nodes
nodes.replace_node(1,2)
print nodes 
