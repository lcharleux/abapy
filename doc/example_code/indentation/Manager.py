from abapy.indentation import Manager, IndentationMesh, Step, RigidCone2D, DeformableCone2D
from abapy.materials import VonMises, Elastic
from math import radians, tan
import numpy as np
import matplotlib.pyplot as plt
#---------------------------------------
# Python post processing function:
# Role: data extraction from odb is performed in Abaqus python but nothing more. A regular Python version featuring numpy/scipy and matplotlib is so much better to perform custom post processing. That's why we do it in two steps: what cannot be done out of abaqus is done in abaqus, everything else is performed outside.
def pypostproc(data):
  if data['completed']:
    return data
  else: 
    print '<Warning: Simulation aborted, check .msg file for explanations.>'
    return data
#---------------------------------------    
# Defining test parameters:
Na, Nb, Ns, Nf = 16,16, 16, 2
half_angle = 70.29
rigid_indenter = False # Sets the indenter rigid of deformable
mesh = IndentationMesh(Na = Na, Nb = Nb, Ns = Ns, Nf = Nf)                # Chosing sample mesh
#indenter = RigidCone2D(half_angle = 70.3)                 # Chosing indenter
indenter = DeformableCone2D(half_angle = half_angle, Na = Na, Nb = Nb, Ns=Ns, Nf=Nf, rigid = rigid_indenter)  
E = 1.                                                    # Young's modulus
sy = E * .01                                              # Yield stress
samplemat = VonMises(labels = 'SAMPLE_MAT', E = E, sy = sy)   # Sample material
indentermat = Elastic(labels = 'INDENTER_MAT', E = E) 
max_disp = .3 * tan(radians(70.3))/tan(radians(half_angle))
nframes = 200
steps = [                                                 # Steps
  Step(name='loading0', nframes = nframes, disp = max_disp/2.),
  Step(name='loading1', nframes = nframes, disp = max_disp), 
  Step(name = 'unloading', nframes = nframes, disp = 0.)] 
#---------------------------------------
# Directories: absolute pathes seems more secure to me since we are running some 'rm'. 
workdir = 'workdir/'
abqlauncher = '/opt/Abaqus/6.9/Commands/abaqus'
simname = 'indentation'
abqpostproc = 'abqpostproc.py'
#---------------------------------------
# Setting simulation manager
m = Manager()
m.set_abqlauncher(abqlauncher)
m.set_workdir(workdir)
m.set_simname(simname)
m.set_abqpostproc(abqpostproc)
m.set_samplemesh(mesh)
m.set_samplemat(samplemat)
m.set_indentermat(indentermat)
m.set_steps(steps)
m.set_indenter(indenter)
m.set_pypostprocfunc(pypostproc)
#---------------------------------------
# Running simulation and post processing
#m.erase_files() # Workdir cleaning
m.make_inp() # INP creation
#m.run_sim() # Running the simulation
#m.run_abqpostproc() # First round of post processing in Abaqus
data = m.run_pypostproc() # Second round of custom post processing in regular Python

#---------------------------------------

if data['completed']:
  # Ploting results
  step2plot = 0
  Nlevels = 200
  plt.figure(0)
  # Ploting load vs. disp curve
  plt.clf()
  ho = data['history']
  F = -ho['force']
  h = -ho['disp']
  C = (F[1]/h[1]**2).average()
  F_fit = C * h **2
  plt.plot(h.plotable()[1], F.plotable()[1], 'b-',label = 'Simulated curve', linewidth = 1.)
  plt.plot(h[0,1].plotable()[1], F_fit[0,1].plotable()[1],'r-', label = 'fitted loading curve', linewidth = 1.)
  plt.xlabel('Displacement $h$')
  plt.ylabel('Force $P$')
  plt.legend()
  plt.grid()
  plt.savefig(workdir + simname + '_load-disp.png')
  # Ploting deformed shape
  
  plt.clf()
  plt.gca().set_aspect('equal')
  plt.axis('off')
  fo = data['field']
  #stress = fo['S'][step2plot].vonmises()
  stress = fo['S'][step2plot].pressure()
  if 'Sind' in fo.keys():
    #ind_stress = fo['Sind'][step2plot].vonmises()
    ind_stress = fo['Sind'][step2plot].pressure()
  smax= max( max(stress.data), max(ind_stress.data))
  smin= min( min(stress.data), min(ind_stress.data)) 
  #levels = [(n+1)/float(Nlevels)*smax for n in xrange(Nlevels)]  
  levels = np.linspace(smin, smax, Nlevels)
  field_flag = r'$\sigma_{eq}$'
  disp = fo['U'][step2plot]
  ind_disp = fo['Uind'][step2plot]
  indenter.apply_displacement(ind_disp) # Applies the displacement to the indenter.
  
  #plt.plot(xbi,ybi,'k-')
  mesh.nodes.apply_displacement(disp) # This Nodes class method allows to deform a Nodes instance (and the Mesh instance it's nested in by the way) using a VectorFieldOutput. This allows very easy mesh tuning and deformed shape ploting.
  xlim, ylim, zlim = mesh.nodes.boundingBox() # This little method allows nicer ploting producing a bounding box with a little gap around the mesh. This avoids the very tight boxes pyplot tends to use which cut thick lines on the border of the mesh.
  xmin, xmax = 0., 2.
  ymin, ymax = -2., 2.
  plt.xlim([xmin, xmax])
  plt.ylim([ymin, ymax])
  x, y, z, tri = mesh.dump2triplot() # This method translates the whole mesh in matplotlib.pyplot.triplot syntax: x coords, y coords, (z coords useless here but can be nice for perspective effects) and label less connectivity.
  xi, yi, zi, trii = indenter.mesh.dump2triplot()
  xe, ye, ze = mesh.get_edges(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)
  xb, yb, zb = mesh.get_border(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)
  xei, yei, zei = indenter.mesh.get_edges(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax) # Gives us a wireframe indenter representation.
  xbi, ybi, zbi = indenter.mesh.get_border(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax) # Gives us the border of the indenter.
  plt.plot(xe,ye,'-k', linewidth = 0.5) # Mesh ploting.
  plt.plot(xb,yb,'-k', linewidth = 1.) # Sample border ploting.
  plt.plot(xei,yei,'-k', linewidth = 0.5) # Mesh ploting.
  plt.plot(xbi,ybi,'-k', linewidth = 1.) # Sample border ploting.
  grad = plt.tricontourf(x, y, tri, stress.data, levels) # Gradiant plot, Nlevels specifies the number of levels.
  plt.tricontourf(xi,yi,trii, ind_stress.data, levels)
  #plt.tricontour(xi,yi,trii, ind_stress.data, levels,  colors = 'black')
  cbar = plt.colorbar(grad)
  cbar.ax.set_ylabel(field_flag, fontsize=20)
  #plt.tricontour(x, y, tri, stress.data, levels,  colors = 'black') # Isovalue plot which make gradiant plot clearer in my (humble) opinion.
  #plt.show()
  plt.savefig(workdir + simname + '_field.png')
