# LOADER: loads simulations to do in the database
import numpy as np
from copy import copy
from abapy.indentation import equivalent_half_angle

# Setting up the database
execfile('settings.py')

# creating some shortcuts
d = db_manager     # database manager
c = db_manager.cls # useful to call Simulation attributs 


#--------------------------
# FIXED PARAMETERS
#--------------------------

# Fixed Parameters  
Na_berk, Nb_berk     = 8, 8  # Mesh parameters
Ns_berk, Nf_berk     = 16, 2 # Mesh parameters
Na_cone, Nb_cone     = 16, 16  # Mesh parameters
Ns_cone, Nf_cone     = 16, 2 # Mesh parameters
Nsweep, sweep_angle = 8, 60. # Mesh sweep parameters
E_s        = 1.   # Sample's Young's modulus
nu         = 0.2   # Poisson's ratio
half_angle = 65.27  # Indenter half angle of the modified Berkovich
frames     = 50    # Number frames per step


#--------------------------
# DRUCKER PRAGER SIMULATIONS
#--------------------------

ey  = [0.01, 0.015,0.02, 0.025,0.03, 0.035,0.04, 0.045, 0.05] # Yield strain
beta = [0., 5., 10., 15., 20., 25., 30.]

print 'LOADING DRUCKER PRAGER SIMULATIONS'

for i in xrange(len(ey)):
  print '* epsilon_y = ', ey[i]
  for j in xrange(len(beta)):
    print 'beta= ', beta[j]
    print '* Conical indenter'
    simu = Simulation( 
      rigid_indenter= True, 
      indenter_pyramid = True,
      three_dimensional = False,
      sweep_angle = sweep_angle,
      sample_mat_type = 'druckerprager',
      sample_mat_args = {'young_modulus': 1., 'poisson_ratio': 0.3, 'yield_stress': ey[i] * E_s, 'beta': beta[j], 'psi':beta[j], 'k': 1. },
      indenter_mat_type = 'elastic',
      indenter_mat_args = {'young_modulus': 1., 'poisson_ratio': 0.3},
      mesh_Na = Na_cone, 
      mesh_Nb = Nb_cone, 
      mesh_Ns = Ns_cone,
      mesh_Nf = Nf_cone, 
      indenter_mesh_Na = 2, 
      indenter_mesh_Nb = 2, 
      indenter_mesh_Ns = 1,
      indenter_mesh_Nf = 1, 
      indenter_mesh_Nsweep = 2,
      mesh_Nsweep = Nsweep,
      indenter_half_angle = equivalent_half_angle(half_angle, sweep_angle),
      frames = frames )
    db_manager.add_simulation(simu)
    
    '''
    print '* Berkovich indenter'
    simu = Simulation( 
      rigid_indenter= True, 
      indenter_pyramid = True,
      three_dimensional = True,
      sweep_angle = sweep_angle,
      sample_mat_type = 'druckerprager',
      sample_mat_args = {'young_modulus': 1., 'poisson_ratio': 0.3, 'yield_stress': ey[i] * E_s, 'beta': beta[j], 'psi':beta[j], 'k': 1. },
      indenter_mat_type = 'elastic',
      indenter_mat_args = {'young_modulus': 1., 'poisson_ratio': 0.3},
      mesh_Na = Na_berk, 
      mesh_Nb = Nb_berk, 
      mesh_Ns = Ns_berk,
      mesh_Nf = Nf_berk, 
      mesh_Nsweep = Nsweep,
      indenter_mesh_Na = 2, 
      indenter_mesh_Nb = 2, 
      indenter_mesh_Ns = 1,
      indenter_mesh_Nf = 2, 
      indenter_mesh_Nsweep = 2,
      indenter_half_angle = half_angle,
      sample_mesh_disp = False,
      frames = frames )
    db_manager.add_simulation(simu)
    '''
  
#--------------------------
# HOLLOMON SIMULATIONS
#--------------------------
#ey  = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01] # Yield strain
ey = [0.001]
#n = [0., .1, .2, .3, .4]
n = [.3, .4]

print 'LOADING HOLLOMON SIMULATIONS'

for i in xrange(len(ey)):
  print '* epsilon_y = ', ey[i]
  for j in xrange(len(n)):
    print '* n = ', n[j]  
    print '* Conical indenter'
    simu = Simulation( 
      rigid_indenter= True, 
      indenter_pyramid = True,
      three_dimensional = False,
      sweep_angle = sweep_angle,
      sample_mat_type = 'hollomon',
      sample_mat_args = {'young_modulus': 1., 'poisson_ratio': 0.3, 'yield_stress': ey[i] * E_s, 'hardening': n[j]},
      indenter_mat_type = 'elastic',
      indenter_mat_args = {'young_modulus': 1., 'poisson_ratio': 0.3},
      mesh_Na = Na_cone, 
      mesh_Nb = Nb_cone, 
      mesh_Ns = Ns_cone,
      mesh_Nf = Nf_cone, 
      indenter_mesh_Na = 2, 
      indenter_mesh_Nb = 2, 
      indenter_mesh_Ns = 1,
      indenter_mesh_Nf = 2, 
      indenter_mesh_Nsweep = 2,
      mesh_Nsweep = Nsweep,
      indenter_half_angle = equivalent_half_angle(half_angle, sweep_angle),
      frames = frames )
    db_manager.add_simulation(simu)
  
  '''
    print '* Berkovich indenter'
    simu = Simulation( 
      rigid_indenter= True, 
      indenter_pyramid = True,
      three_dimensional = True,
      sweep_angle = sweep_angle,
      sample_mat_type = 'hollomon',
      sample_mat_args = {'young_modulus': 1., 'poisson_ratio': 0.3, 'yield_stress': ey[i] * E_s, 'hardening': n[j]},
      indenter_mat_type = 'elastic',
      indenter_mat_args = {'young_modulus': 1., 'poisson_ratio': 0.3},
      mesh_Na = Na_berk, 
      mesh_Nb = Nb_berk, 
      mesh_Ns = Ns_berk,
      mesh_Nf = Nf_berk, 
      mesh_Nsweep = Nsweep,
      indenter_mesh_Na = 2, 
      indenter_mesh_Nb = 2, 
      indenter_mesh_Ns = 1,
      indenter_mesh_Nf = 2, 
      indenter_mesh_Nsweep = 2,
      indenter_half_angle = half_angle,
      sample_mesh_disp = False,
      frames = frames )
    db_manager.add_simulation(simu)
   '''
   

