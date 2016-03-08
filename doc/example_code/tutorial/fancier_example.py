# FANCIER_EXAMPLE: computing the loading prefactor of the indentation load vs. disp curve.
# Run using python

# Packages
from abapy.misc import load
import matplotlib.pyplot as plt
import numpy as np

     


def compute_C(E = 1. , nu = 0.3, sy = 0.01, abqlauncher = '/opt/Abaqus/6.9/Commands/abaqus', workdir = 'workdir', name = 'indentation_axi_fancier', frames = 50):
  '''
  Computes the load prefactor C using Abaqus.
  
  Inputs:
  * E: sample's Young modulus.
  * nu: samples's Poisson's ratio
  * sy: sample's yield stress (von Mises yield criterion).
  * abqlauncher: absolute path to abaqus launcher.
  * wordir: path to working directory, can be relative.
  * name: name of simulation files.
  * frames: number of frames per step, increase if the simulation does not complete.
  
  Returns:
  * Load prefactor C (float)
  '''
  import time, subprocess, os
  t0 = time.time() # Starting time recording
  path = workdir + '/' + name
  # Reading the INP target file
  f = open('indentation_axi_target.inp', 'r')
  inp = f.read()
  f.close()
  # Replace the targets in the file
  inp = inp.replace('#E', '{0}'.format(E))
  inp = inp.replace('#NU', '{0}'.format(nu))
  inp = inp.replace('#SY', '{0}'.format(sy))
  inp = inp.replace('#FRAME', '{0}'.format(1./frames))
  # Creating a new inp file
  f = open(path  + '.inp', 'w')
  f.write(inp)
  f.close()
  print 'Created INP file: {0}.inp'.format(path)
  # Then we run the simulation
  print 'Running simulation in Abaqus'
  p = subprocess.Popen( '{0} job={1} input={1}.inp interactive ask_delete=OFF'.format(abqlauncher, name), cwd = workdir, shell=True, stdout = subprocess.PIPE)
  trash = p.communicate()
  
  # Now we test run the post processing script
  print 'Post processing the simulation in Abaqus/Python'
  p = subprocess.Popen( [abqlauncher,  'viewer', 'noGUI=fancier_example_abq.py'], cwd = '.',stdout = subprocess.PIPE )
  trash = p.communicate()
  # Getting back raw data
  data = load(workdir + '/' + name + '.pckl')
  # Post processing
  print 'Post processing the simulation in Python'
  if data['completed']:
    ref_node_label = data['ref_node_label']
    force_hist = -data['RF2']['Node I_INDENTER.{0}'.format(ref_node_label)]
    disp_hist = -data['U2']['Node I_INDENTER.{0}'.format(ref_node_label)]
    trash, force_l = force_hist[0,1].plotable() # Getting back force  during loading
    trash, disp_l = disp_hist[0,1].plotable()   # Getting backdisplacement during loading
    trash, force_u = force_hist[2].plotable() # Getting back force  during unloading
    trash, disp_u = disp_hist[2].plotable()   # Getting backdisplacement during unloading
    C_factor = (force_hist[1] / disp_hist[1]**2).average()
    
  else:
    print 'Simulation aborted, probably because frame number is to low'
    C_factor = None  
  t1 = time.time()
  print 'Time used: {0:.2e} s'.format(t1-t0)
  return C_factor
  
# Setting up some pathes
workdir = 'workdir'
name = 'indentation_axi_fancier'
abqlauncher = '/opt/Abaqus/6.9/Commands/abaqus'

# Setting material parameters
E = 1.
nu = 0.3
sy = 0.01
frames = 50

# Testing it all
C = compute_C(E = E, nu = nu, sy = sy, frames = frames)
print 'C = {0}'.format(C)
