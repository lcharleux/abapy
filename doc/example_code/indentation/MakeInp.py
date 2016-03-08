#-----------------------
# PACKAGES
from abapy.indentation import MakeInp, DeformableCone2D, DeformableCone3D, IndentationMesh, Step
from abapy.materials import Elastic, VonMises
from math import radians, tan
#-----------------------

#-----------------------
# FIRST EXEMPLE: AXISYMMETRIC INDENTATION
# Model parameters
half_angle = 70.29      # Indenter half angle, 70.3 degrees is equivalent to Berkovich indenter
Na, Nb, Ns, Nf, l = 8, 8, 16, 2, 1.
E, nu = 1., 0.3         # E is the Young's modulus and nu is the Poisson's ratio. 
ey = 0.01 * E
max_disp = l/3.*tan(radians(70.3))/tan(radians(half_angle)) # Maximum displacement

# Building model 
indenter = DeformableCone2D(half_angle = half_angle, rigid = True, Na = Na, Nb = Nb, Ns = Ns, Nf = Nf, l=l) # Indenter
sample_mesh = IndentationMesh(Na = Na, Nb = Nb, Ns = Ns, Nf = Nf, l=l)# Sample Mesh
#sample_mat = Elastic(labels = 'SAMPLE_MAT', E = E, nu=  nu)    # Sample material
sample_mat = VonMises(labels = 'SAMPLE_MAT', E = E, nu=  nu, sy = E * ey)    # Sample material
indenter_mat = Elastic(labels = 'INDENTER_MAT')    # Indenter material
steps = [                                                 # Steps
  Step(name='preloading', nframes = 50, disp = max_disp / 2.),
  Step(name='loading',    nframes = 50, disp = max_disp ), 
  Step(name='unloading',  nframes = 50, disp = 0.), ] 

# Making INP file
inp = MakeInp(indenter = indenter,
sample_mesh = sample_mesh,
sample_mat = sample_mat,
indenter_mat = indenter_mat,
steps = steps)
f = open('workdir/indentation_axi.inp', 'w')
f.write(inp)
f.close() 
#-----------------------

#-----------------------
# SECOND EXAMPLE: 3D BERKOVICH INDENTATION
# Model Parameters
half_angle = 65.27     # Indenter half angle, 65.27 leads to a modified Berkovich geometry, see help(DeformableCone3D for further information)
Na, Nb, Ns, Nf, l = 8, 8, 8, 2, 1. # with 4, 4, 4, 2, 1., simulation is very fast, the mesh is coarse (even crappy) but the result is surprisingly good! 
sweep_angle, N = 60., 8
E, nu = 1., 0.3         # E is the Young's modulus and nu is the Poisson's ratio. 
ey = 0.01 # yield strain
max_disp = l/3.*tan(radians(70.3))/tan(radians(half_angle)) # Maximum displacement
pyramid = True

# Building model
indenter = DeformableCone3D(half_angle = half_angle, rigid = True, Na = Na, Nb = Nb, Ns = Ns, Nf = Nf, N= N, sweep_angle=sweep_angle, l=l, pyramid = pyramid) # Indenter
sample_mesh = IndentationMesh(Na = Na, Nb = Nb, Ns = Ns, Nf = Nf, l=l).sweep(sweep_angle = sweep_angle, N = N)   # Sample Mesh
#sample_mat = Elastic(labels = 'SAMPLE_MAT', E = E, nu=  nu)    # Sample material
sample_mat = VonMises(labels = 'SAMPLE_MAT', E = E, nu=  nu, sy = E * ey)    # Sample material
indenter_mat = Elastic(labels = 'INDENTER_MAT')    # Indenter material
steps = [                                                 # Steps
  Step(name='preloading',   nframes = 100, disp = max_disp / 2., boundaries_3D=True), 
  Step(name='loading',      nframes = 100, disp = max_disp,      boundaries_3D=True), 
  Step(name='unloading',    nframes = 200, disp = 0.,            boundaries_3D=True)] 

# Making INP file.
inp = MakeInp(indenter = indenter,
sample_mesh = sample_mesh,
sample_mat = sample_mat,
indenter_mat = indenter_mat,
steps = steps, is_3D = True)
f = open('workdir/indentation_berko.inp', 'w')
f.write(inp)
f.close() 
#-----------------------

# Simulation can be launched using abaqus job=indentation and can be roughly post processed using abaqus viewer of more finely using abapy.
