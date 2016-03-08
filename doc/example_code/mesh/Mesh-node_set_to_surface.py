from abapy import mesh

m = mesh.RegularQuadMesh(N1 = 4, N2 = 4, l1 = 2., l2 = 6.)
m.nodes.sets = {}
m.nodes.add_set_by_func("top_nodes", lambda x,y,z,labels : y == 6.)
m.node_set_to_surface("top_surface", "top_nodes")
