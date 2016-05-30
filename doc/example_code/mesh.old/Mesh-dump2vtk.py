from abapy.mesh import RegularQuadMesh
from abapy.indentation import IndentationMesh
import numpy as np

def tensor_function(x, y, z, labels):
  """
  Vector function used to produced the displacement field.
  """
  r0 = 1.
  theta = .5 * np.pi * x
  r = y + r0
  s11 = z + x
  s22 = z + y
  s33 = x**2
  s12 = y**2
  s13 = x + y 
  s23 = z
  return s11, s22, s33, s12, s13, s23 

def vector_function(x, y, z, labels):
  """
  Vector function used to produced the displacement field.
  """
  r0 = 1.
  theta = .5 * np.pi * x
  r = y + r0
  ux = -x + r * np.cos(theta**2)
  uy = -y + r * np.sin(theta**2)
  uz = 0. * z
  return ux, uy, uz

def scalar_function(x, y, z, labels):
  """
  Scalar function used to produced the plotted field.
  """
  return x**2 + y**2
#MESH GENERATION
N1, N2, N3 = 8, 8, 8
l1, l2, l3 = .75, 1., .5
m = RegularQuadMesh(N1 = N1, N2 = N2, l1 = l1, l2 = l2).extrude(N = N3, l = l3)
#FIELDS GENERATION
s = m.nodes.eval_tensorFunction(tensor_function)
m.add_field(s, "s")
u = m.nodes.eval_vectorFunction(vector_function)
m.add_field(u, "u")
m.nodes.apply_displacement(u)
f = m.nodes.eval_function(scalar_function)
m.add_field(f, "f")
m.dump2vtk("Mesh-dump2vtk.vtk")


