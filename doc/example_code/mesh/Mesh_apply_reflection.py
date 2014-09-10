from abapy.mesh import RegularQuadMesh
from abapy.indentation import ParamInfiniteMesh
import matplotlib.pyplot as plt

point  = (0., 0., 0.)
normal = (1., 0., 0.)
m0 = ParamInfiniteMesh()
x0, y0, z0 = m0.get_edges()
m1 = m0.apply_reflection(normal = normal, point = point)
x1, y1, z1 = m1.get_edges()
plt.plot(x0, y0)
plt.plot(x1, y1)
plt.gca().set_aspect('equal')
plt.show()

