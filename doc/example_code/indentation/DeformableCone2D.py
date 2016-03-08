from abapy.indentation import DeformableCone2D
from matplotlib import pyplot as plt
c = DeformableCone2D(Na =8, Nb = 8, Ns = 1)
f = open('DeformableCone2D.inp', 'w')
f.write(c.dump2inp())
f.close()
x,y,z = c.mesh.get_edges()
plt.plot(x,y)
plt.gca().set_aspect('equal')
plt.show()

