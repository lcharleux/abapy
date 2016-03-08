from abapy.misc import load
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import mpl
import numpy as np

path_to_odb = '../../../../testing/'
title0 = 'title0'
title1 = 'title1'
title2 = 'title2'
N_levels = 10 # Number os isovalues
levels = np.linspace(0., 0.3, N_levels)

S = load(path_to_odb + 'indentation_S.pckl')
mesh0 = load(path_to_odb + 'indentation_mesh.pckl')
#mesh0 = mesh0['core_1']
point  = (0., 0., 0.)
normal = (1., 0., 0.)
mesh1 = mesh0.apply_reflection(normal = normal, point = point)
S = S[mesh0.nodes.labels.tolist()]
x0, y0, z0, tri0 = mesh0.dump2triplot()
xe0, ye0, ze0 = mesh0.get_edges()
xb0, yb0, zb0 = mesh0.get_border()
xlim0, ylim0, zlim0 = mesh0.nodes.boundingBox()
x1, y1, z1, tri1 = mesh1.dump2triplot()
xe1, ye1, ze1 = mesh1.get_edges()
xb1, yb1, zb1 = mesh1.get_border()
xlim2, ylim1, zlim1 = mesh1.nodes.boundingBox()


field0 = S.get_component(12) # What to plot ?
field1 = field0 # What other field to plot ?

fig = plt.figure(0)
plt.clf()

sp0 = plt.subplot(122)
sp1 = plt.subplot(121)
sp0.set_title(title0)
sp1.set_title(title1)
sp0.set_xlabel('$r$', fontsize =15.)
sp1.set_xlabel('$r$', fontsize =15.)
sp1.set_ylabel('$z$', fontsize =15.)

sp0.set_xticks([])
sp0.set_yticks([])
sp1.set_xticks([])
sp1.set_yticks([])
sp0.set_aspect('equal')
sp1.set_aspect('equal')
sp0.plot(xe0,ye0, 'k-', linewidth = .5) # edge ploting 
sp0.plot(xb0,yb0, 'k-', linewidth = .5) # edge ploting 
sp1.plot(xe1,ye1, 'k-', linewidth = .5) # edge ploting 
sp1.plot(xb1,yb1, 'k-', linewidth = .5) # edge ploting 

sp0.tricontourf(x0,y0,tri0,field0.data, levels = levels)
cs0 = sp0.tricontour(x0,y0,tri0,field0.data, levels = levels, colors = 'black')
sp1.tricontourf(x1,y1,tri1,field1.data, levels = levels)
cs1 = sp1.tricontour(x1,y1,tri1,field1.data, levels = levels, colors = 'black')
plt.clabel(cs1, fmt = '%2.2f', colors = 'w', fontsize=14)
plt.clabel(cs0, fmt = '%2.2f', colors = 'w', fontsize=14)

plt.show()
