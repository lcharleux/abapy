from abapy.mesh import Mesh, parseInp, writeInp, writeMsh, Node, Tri3, Quad4
import numpy as np
import os

m = Mesh()
m.nodes[1] = Node( (0., 1., 0.) )
m.nodes[2] = Node( (0., 2., 0.) )
m.nodes[3] = Node( (0., 3., 0.) )
m.nodes[4] = Node( (1., 4., 0.) )
m.nodes[5] = Node( (1., 3., 0.) )
m.nodes[6] = Node( (1., 2., 0.) )
m.nodes[7] = Node( (2., 1., 0.) )
m.nodes[8] = Node( (1., 0., 0.) )
m.nodes[9] = Node( (2., 3., 0.) )
m.nodes[10] = Node( (2., 2., 0.) )

m.elements[1] = Quad4(conn = (1,8,7,6))
m.elements[2] = Quad4(conn = (2,6,5,3))
m.elements[3] = Tri3(conn = (1,6,2))
m.elements[4] = Tri3(conn = (3,5,4))
m.elements[5] = Tri3(conn = (6,7,10))
m.elements[6] = Quad4(conn = (5,6,10,9))

[n.sets.add("this_set") for n in m.nodes.values() if n.coords[0] == 1 ]
m.node_set_to_surface("this_set", "that_surface")
print m
