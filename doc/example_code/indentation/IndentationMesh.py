from abapy.indentation import IndentationMesh
from matplotlib import pyplot as plt

Na = 16 # Elements along x axis in the square zone
Nb = 16 # Elements along y axis in the square zone
Ns = 2 # Radial number of elements in the shell
Nf = 2 # Minimal number of orthoradial elements in heach half shell
l = 1. # Size of the square zone
name = 'CAX4' # Name of the elements  

m = IndentationMesh(Na = Na, Nb = Nb, Ns = Ns, Nf= Nf, l = l, name = name)

m_core = m['core_elements']
m_shell = m['shell_elements']

x_core, y_core, z_core = m_core.get_edges()
x_shell, y_shell, z_shell = m_shell.get_edges()
plt.figure(0)
plt.clf()
plt.axis('off')
plt.grid()
xlim, ylim, zlim = m.nodes.boundingBox()
plt.gca().set_aspect('equal')
plt.xlim(xlim)
plt.ylim(ylim)
plt.plot(x_core,y_core, 'b-', label = 'Core elements')
plt.plot(x_shell,y_shell, 'r-', label = 'Shell elements')
plt.legend()
plt.show()
