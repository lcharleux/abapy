from abapy.future import Mesh, parseInp, writeInp, writeMsh, Node, Tri3, Quad4
import numpy as np

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

m = m.extrude(translation = [0, 0, 1], layers = 1)

def transformation(x, y, z):
  theta =  np.pi / 2.  * z
  r = x.copy() 
  x = r * np.cos(theta)
  z = r * np.sin(theta)
  return x, y, z

m = m.transform(transformation)
mapping = m.overlapping_nodes()
m = m.merge_nodes(mapping)
m2 = m.copy()
"""
m2.elements.clear()
counter = 1
for label in m.elements.keys():
  el = m.elements[label]
  simplices = el.simplex_decomposition()
  for simplex in simplices:
    m2.elements[cunter] = simplex
    counter +=1
"""    
m2 = m2.clean_connectivity()


writeInp(m2, path = "clean_connectivity.inp", mapping = {"Tri3": "CAX3", "Quad4": "CAX4", "Hexa8": "C3D8", "Prism6":"C3D6", "Tetra4": "C3D4"})
writeMsh(m2, path = "clean_connectivity.msh")

