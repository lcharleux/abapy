# VON MISES CLASSES
# Parametric indentation tool

#----------------------------------------------------------------------------------------------------------------
# IMPORTS
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, PickleType, UniqueConstraint, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
#----------------------------------------------------------------------------------------------------------------


Base = declarative_base()




#---------------------------------------------------------------------------------------------------------------- 
# SIMULATION CLASS 
# Declaration of the Simulation class which stores all simulation data so that each simulation is an instance of the Simulation class. This class uses widely SQL Alchemy's ORM capabilities. To go deeper into this class, please read the (very good) tutorial or SQL Alchemy. If you just want to see the spirit of this class, just see it as a database when each attribute is mirrored as a database entry. The structure of the class itself is surprising because SQL Alchemy allows automatic constructor building so any attribute is automatically available in the constructor (i. e. __init__).
class Simulation(Base):
  __tablename__ = 'simulations'
  id = Column(Integer, primary_key=True)
  # Inputs
  three_dimensional            = Column(Boolean, default = False, nullable = False)
  sweep_angle         = Column(Float, nullable = False, default = 60.)
  rigid_indenter               = Column(Boolean, default = True, nullable = False)
  indenter_half_angle          = Column(Float, nullable = False, default = 70.3)
  indenter_pyramid             = Column(Boolean, default = True, nullable = False)
  indenter_mat_type            = Column(String, nullable = False, default = 'elastic')
  indenter_mat_args            = Column(PickleType, 
    nullable = False, 
    default = {'young_modulus': 1., 'poisson_ratio': 0.3})
  sample_mat_type              = Column(String, nullable = False, default = 'vonmises')
  sample_mat_args              = Column(PickleType, 
    nullable = False, 
    default = {'young_modulus': 1., 'poisson_ratio': 0.3, 'yield_stress': 0.01})
  friction                     = Column(Float, nullable = False, default = 0.) 
  mesh_Na                      = Column(Integer, default = 4,  nullable = False)
  mesh_Nb                      = Column(Integer, default = 4,  nullable = False)
  mesh_Ns                      = Column(Integer, default = 16, nullable = False)
  mesh_Nf                      = Column(Integer, default = 2,  nullable = False)
  mesh_Nsweep                  = Column(Integer, default = 8,  nullable = False)
  indenter_mesh_Na             = Column(Integer, default = 0,  nullable = False)
  indenter_mesh_Nb             = Column(Integer, default = 0,  nullable = False)
  indenter_mesh_Ns             = Column(Integer, default = 0,  nullable = False)
  indenter_mesh_Nf             = Column(Integer, default = 0,  nullable = False)
  indenter_mesh_Nsweep         = Column(Integer, default = 0,  nullable = False)
  mesh_l                       = Column(Float,   default = 1., nullable = False)
  max_disp                     = Column(Float,   default = 1., nullable = False)
  sample_mesh_disp             = Column(PickleType, nullable = False, default = False )
  # Internal parameters
  frames                       = Column(Integer, default = 30, nullable = False)
  completed                    = Column(Boolean, default = False, nullable = False)
  priority                     = Column(Integer, default = 1, nullable = False)
  # Preprocess
  mesh                         = Column(PickleType)
  indenter                     = Column(PickleType)
  sample_mat                   = Column(PickleType)
  indenter_mat                 = Column(PickleType)
  steps                        = Column(PickleType)
  # Time histories
  force_hist                   = Column(PickleType)
  disp_hist                    = Column(PickleType)
  tip_penetration_hist         = Column(PickleType)
  elastic_work_hist            = Column(PickleType)
  plastic_work_hist            = Column(PickleType)
  friction_work_hist           = Column(PickleType)
  total_work_hist              = Column(PickleType)
  # Contact data
  contact_data                 =  Column(PickleType)
  # Fields
  stress_field                 = Column(PickleType)
  disp_field                   = Column(PickleType)
  total_strain_field           = Column(PickleType)
  plastic_strain_field         = Column(PickleType)
  elastic_strain_field         = Column(PickleType)
  equivalent_plastic_strain_field = Column(PickleType)
  disp_field                   = Column(PickleType)
  ind_disp_field               = Column(PickleType)
  
  # Table args 
  __table_args__ = (
    UniqueConstraint(
    'three_dimensional',
    'indenter_pyramid',
    'rigid_indenter',
    'sample_mat_type',
    'sample_mat_args',
    'indenter_mat_type',
    'indenter_mat_args',
    'indenter_half_angle',
    'sweep_angle',
    'friction',
    'mesh_Na',
    'mesh_Nb', 
    'mesh_Ns', 
    'mesh_Nf',
    'mesh_l',
    'mesh_Nsweep',
    'indenter_mesh_Na',
    'indenter_mesh_Nb', 
    'indenter_mesh_Ns', 
    'indenter_mesh_Nf',
    'indenter_mesh_Nsweep',
    'max_disp',
    'sample_mesh_disp'),
    {})
  
  # Post processing script
  def abqpostproc_byRpt(self):
    if self.three_dimensional: # 3D post processing script
      out = """# ABQPOSTPROC.PY
# Warning: executable only in abaqus abaqus viewer -noGUI,... not regular python.
import sys
from abapy.postproc import GetFieldOutput_byRpt as gfo
from abapy.postproc import GetVectorFieldOutput_byRpt as gvfo
from abapy.postproc import GetTensorFieldOutput_byRpt as gtfo
from abapy.postproc import GetHistoryOutputByKey as gho
from abapy.indentation import Get_ContactData
from abapy.misc import dump
from odbAccess import openOdb
from abaqusConstants import JOB_STATUS_COMPLETED_SUCCESSFULLY



# Odb opening  
file_name = '#FILE_NAME'
odb = openOdb(file_name + '.odb')
data = {}

# Check job status:
job_status = odb.diagnosticData.jobStatus

if job_status == JOB_STATUS_COMPLETED_SUCCESSFULLY:
  data['completed'] = True 
  # Field Outputs
  data['field'] = {}
  fo = data['field']
  fo['Uind'] = [
    gvfo(odb = odb, 
      instance = 'I_INDENTER', 
      step = 1,
      frame = -1,
      original_position = 'NODAL', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'U', 
      delete_report = True),
    gvfo(odb = odb, 
      instance = 'I_INDENTER', 
      step = 2,
      frame = -1,
      original_position = 'NODAL', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'U', 
      delete_report = True)]
  
  fo['U'] = [
    gvfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'NODAL', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'U', 
      sub_set_type = 'element', 
      delete_report = True),
    gvfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'NODAL', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'U', 
      sub_set_type = 'element', 
      delete_report = True)]
      
  fo['S'] = [
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'S', 
      sub_set_type = 'element', 
      delete_report = True),
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'S', 
      sub_set_type = 'element', 
      delete_report = True)]
   
  fo['LE'] = [
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'LE', 
      sub_set_type = 'element', 
      delete_report = True),
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'LE', 
      sub_set_type = 'element', 
      delete_report = True)] 
      
  fo['EE'] = [
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'EE', 
      sub_set_type = 'element', 
      delete_report = True),
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'EE', 
      sub_set_type = 'element', 
      delete_report = True)]     
  
  fo['PE'] = [
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'PE', 
      sub_set_type = 'element', 
      delete_report = True),
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'PE', 
      sub_set_type = 'element', 
      delete_report = True)] 
  
  fo['PEEQ'] = [
    gfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'PEEQ', 
      sub_set_type = 'element', 
      delete_report = True),
    gfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'PEEQ', 
      sub_set_type = 'element', 
      delete_report = True)] 
  # History Outputs
  data['history'] = {} 
  ho = data['history']
  ref_node = odb.rootAssembly.instances['I_INDENTER'].nodeSets['REF_NODE'].nodes[0].label
  ho['force'] =  gho(odb,'RF2')['Node I_INDENTER.'+str(ref_node)] # GetFieldOutputByKey returns all the occurences of the required output (here 'RF2') and stores it in a dict. Each dict key refers to a location. Here we have to specify the location ('Node I_INDENTER.1') mainly for displacement which has been requested at several locations.
  ho['disp'] =   gho(odb,'U2')['Node I_INDENTER.'+str(ref_node)]
  tip_node = odb.rootAssembly.instances['I_INDENTER'].nodeSets['TIP_NODE'].nodes[0].label
  ho['tip_penetration'] =   gho(odb,'U2')['Node I_INDENTER.'+str(tip_node)]
  ho['allse'] =   gho(odb,'ALLSE').values()[0]
  ho['allpd'] =   gho(odb,'ALLPD').values()[0]
  ho['allfd'] =   gho(odb,'ALLFD').values()[0]
  ho['allwk'] =   gho(odb,'ALLWK').values()[0]
  #ho['carea'] =  gho(odb,'CAREA    ASSEMBLY_I_SAMPLE_SURFACE_FACES/ASSEMBLY_I_INDENTER_SURFACE_FACES').values()[0]
  
  # CONTACT DATA PROCESSING
  ho['contact'] = Get_ContactData(odb = odb, instance = 'I_SAMPLE', node_set = 'TOP_NODES')
 
else:
  data['completed'] = False
# Closing and dumping
odb.close()
dump(data, file_name+'.pckl')"""
    else:
      out = """# ABQPOSTPROC.PY
# Warning: executable only in abaqus abaqus viewer -noGUI,... not regular python.
import sys
from abapy.postproc import GetFieldOutput_byRpt as gfo
from abapy.postproc import GetVectorFieldOutput_byRpt as gvfo
from abapy.postproc import GetTensorFieldOutput_byRpt as gtfo
from abapy.postproc import GetHistoryOutputByKey as gho
from abapy.indentation import Get_ContactData
from abapy.misc import dump
from odbAccess import openOdb
from abaqusConstants import JOB_STATUS_COMPLETED_SUCCESSFULLY



# Odb opening  
file_name = '#FILE_NAME'
odb = openOdb(file_name + '.odb')
data = {}

# Check job status:
job_status = odb.diagnosticData.jobStatus

if job_status == JOB_STATUS_COMPLETED_SUCCESSFULLY:
  data['completed'] = True 
  # Field Outputs
  data['field'] = {}
  fo = data['field']
  fo['Uind'] = [
    gvfo(odb = odb, 
      instance = 'I_INDENTER', 
      step = 1,
      frame = -1,
      original_position = 'NODAL', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'U', 
      delete_report = True),
    gvfo(odb = odb, 
      instance = 'I_INDENTER', 
      step = 2,
      frame = -1,
      original_position = 'NODAL', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'U', 
      delete_report = True)]
  
  fo['U'] = [
    gvfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'NODAL', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'U', 
      sub_set_type = 'element', 
      delete_report = True),
    gvfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'NODAL', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'U', 
      sub_set_type = 'element', 
      delete_report = True)]
      
  fo['S'] = [
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'S', 
      sub_set_type = 'element', 
      delete_report = True),
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'S', 
      delete_report = True)]
   
  fo['LE'] = [
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'LE', 
      sub_set_type = 'element', 
      delete_report = True),
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'LE', 
      sub_set_type = 'element', 
      delete_report = True)] 
      
  fo['EE'] = [
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'EE', 
      sub_set_type = 'element', 
      delete_report = True),
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'EE', 
      sub_set_type = 'element', 
      delete_report = True)]     
  
  fo['PE'] = [
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'PE', 
      sub_set_type = 'element', 
      delete_report = True),
    gtfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'PE', 
      sub_set_type = 'element', 
      delete_report = True)] 
  
  fo['PEEQ'] = [
    gfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 1,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'PEEQ', 
      sub_set_type = 'element', 
      delete_report = True),
    gfo(odb = odb, 
      instance = 'I_SAMPLE', 
      step = 2,
      frame = -1,
      original_position = 'INTEGRATION_POINT', 
      new_position = 'NODAL', 
      position = 'node',
      field = 'PEEQ', 
      sub_set_type = 'element', 
      delete_report = True)] 
  # History Outputs
  data['history'] = {} 
  ho = data['history']
  ref_node = odb.rootAssembly.instances['I_INDENTER'].nodeSets['REF_NODE'].nodes[0].label
  ho['force'] =  gho(odb,'RF2')['Node I_INDENTER.'+str(ref_node)] # GetFieldOutputByKey returns all the occurences of the required output (here 'RF2') and stores it in a dict. Each dict key refers to a location. Here we have to specify the location ('Node I_INDENTER.1') mainly for displacement which has been requested at several locations.
  ho['disp'] =   gho(odb,'U2')['Node I_INDENTER.'+str(ref_node)]
  tip_node = odb.rootAssembly.instances['I_INDENTER'].nodeSets['TIP_NODE'].nodes[0].label
  ho['tip_penetration'] =   gho(odb,'U2')['Node I_INDENTER.'+str(tip_node)]
  ho['allse'] =   gho(odb,'ALLSE').values()[0]
  ho['allpd'] =   gho(odb,'ALLPD').values()[0]
  ho['allfd'] =   gho(odb,'ALLFD').values()[0]
  ho['allwk'] =   gho(odb,'ALLWK').values()[0]
  #ho['carea'] =  gho(odb,'CAREA    ASSEMBLY_I_SAMPLE_SURFACE_FACES/ASSEMBLY_I_INDENTER_SURFACE_FACES').values()[0]
  
  # CONTACT DATA PROCESSING
  ho['contact'] = Get_ContactData(odb = odb, instance = 'I_SAMPLE', node_set = 'TOP_NODES')
 
else:
  data['completed'] = False
# Closing and dumping
odb.close()
dump(data, file_name+'.pckl')"""
    return out
  # Scalar Outputs
  # Load Prefactor
  load_prefactor = Column(Float)
  def Load_prefactor(self, update = False):
    '''
    Defined during loading phase by force = c * penetration **2 where c is the load prefactor.
    '''
    load_prefactor = (self.force_hist[1] / self.disp_hist[1]**2).average(method = 'simps')
    if update: 
      self.load_prefactor = load_prefactor
    else: 
      return load_prefactor
 
  # Irreversible work ratio:
  irreversible_work_ratio = Column(Float)
  def Irreversible_work_ratio(self, update = False):
    '''
    Irreversible work divided by the total work
    '''
    unload = self.total_work_hist[2]
    irreversible_work_ratio = 3* unload.data_min() / self.load_prefactor
    if update: 
      self.irreversible_work_ratio = irreversible_work_ratio
    else: 
      return irreversible_work_ratio
  
  # Plastic work ratio:
  plastic_work_ratio = Column(Float)
  def Plastic_work_ratio(self, update = False):
    '''
    Plastic work divided by the total work
    '''
    plastic_work_ratio = (self.plastic_work_hist[1] / self.total_work_hist[1]).average()
    if update: 
      self.plastic_work_ratio = plastic_work_ratio
    else: 
      return plastic_work_ratio
  
  # Unloading fit
  contact_stiffness = Column(Float)
  final_displacement = Column(Float)
  unload_exponent = Column(Float)
  def Unloading_fit(self):
    '''
    Computes the contact stiffness and the final displacement using a power fit of the unloading curve between 0% and 100% of the max force. The final penetration is divided by the maximum penetration and the contact stiffness is multiplied by the ratio of the max displacement by the max force.
    '''
    import numpy as np
    from scipy.optimize import leastsq
    unload = 2
    disp = np.array(self.disp_hist[unload].data[0])
    force = np.array(self.force_hist[unload].data[0])
    max_force = force.max()
    max_disp = disp.max()
    loc = np.where(force >= max_force * .1)
    disp = disp[loc] /max_disp
    force = force[loc] / max_force
    func = lambda k, x: ( (x - k[0]) / (1. - k[0] ) )**k[1] 
    err = lambda v, x, y: (func(v,x)-y)
    k0 = [0., 1.]
    k, success = leastsq(err, k0, args=(disp,force), maxfev=10000)
    self.final_displacement = k[0] / max_disp
    self.contact_stiffness = k[1] / (1. - k[0]) 
    self.unload_exponent = k[1] 
       
  # Contact area:
  contact_area = Column(Float)
  def Contact_area(self, update = False):
    '''
    Cross contact areat under load divided by the square of the indenter displacement.
    '''
    import numpy as np
    from abapy.postproc import HistoryOutput
    contact_step = self.contact_data[1] # we only look at the loading 1 here
    ca= np.array([contact_step[i].contact_area() for i in xrange(len(contact_step))])
    disp = np.array(self.disp_hist[1].data)
    ca = ca / (disp**2)
    contact_area = ca.mean()
    if update: 
      self.contact_area = contact_area
    else: 
      return contact_area
  
  
  # Tip penetration    
  tip_penetration = Column(PickleType)
  def Tip_penetration(self, update = True):
    '''
    Tip penetration under load divided by the displacement.
    '''
    tip_pen = self.tip_penetration_hist[1]
    disp = self.disp_hist[1]
    tip_pen = (-tip_pen/disp).average()
    if update: 
      self.tip_penetration = tip_pen
    else: 
      return tip_penetration
  
  
  
 
  def __repr__(self):
    return '<Simulation: id={0}>'.format(self.id)
  
  def difficulty(self):
    if self.sample_mat_type == 'elastic': mat_diff = 1.
    if self.sample_mat_type == 'vonmises': 
      args = self.sample_mat_args
      mat_diff = args['young_modulus'] / args['yield_stress']
    if self.sample_mat_type == 'druckerprager': 
      args = self.sample_mat_args
      mat_diff = args['young_modulus'] / args['yield_stress']
    if self.sample_mat_type == 'hollomon': 
      args = self.sample_mat_args
      mat_diff = args['young_modulus'] / args['yield_stress']
    mesh_diff = self.max_disp /self.mesh_l * (self.mesh_Na + self.mesh_Nb) / 2.
    return mat_diff * mesh_diff
  
  def preprocess(self):
    from abapy.indentation import IndentationMesh, Step, DeformableCone2D, DeformableCone3D
    from abapy.materials import VonMises, Elastic, DruckerPrager, Hollomon
    from math import tan, radians
    mesh_l = 2 * max( self.max_disp , tan(radians(self.indenter_half_angle)) ) # Adjusting mesh size to max_disp
    if self.three_dimensional:
      self.mesh = IndentationMesh(
        Na = self.mesh_Na, 
        Nb = self.mesh_Nb, 
        Ns = self.mesh_Ns, 
        Nf = self.mesh_Nf, 
        l = mesh_l).sweep(
        sweep_angle = self.sweep_angle,
        N = self.mesh_Nsweep)
      if self.sample_mesh_disp != False:
        field = self.mesh.nodes.eval_vectorFunction(self.sample_mesh_disp)
        self.mesh.nodes.apply_displacement(field)
      if self.indenter_mesh_Nf == 0: Nf_i = self.mesh_Nf 
      self.indenter = DeformableCone3D(
        half_angle = self.indenter_half_angle, 
        sweep_angle = self.sweep_angle,
        pyramid = self.indenter_pyramid,
        l = mesh_l, 
        Na = self.mesh_Na * (self.indenter_mesh_Na == 0) + self.indenter_mesh_Na * (self.indenter_mesh_Na != 0), 
        Nb = self.mesh_Nb * (self.indenter_mesh_Nb == 0) + self.indenter_mesh_Nb * (self.indenter_mesh_Nb != 0), 
        Ns = self.mesh_Ns * (self.indenter_mesh_Ns == 0) + self.indenter_mesh_Ns * (self.indenter_mesh_Ns != 0), 
        Nf = self.mesh_Nf * (self.indenter_mesh_Nf == 0) + self.indenter_mesh_Nf * (self.indenter_mesh_Nf != 0), 
        N =  self.mesh_Nsweep * (self.indenter_mesh_Nsweep == 0) + self.indenter_mesh_Nsweep * (self.indenter_mesh_Nsweep != 0),
        rigid = self.rigid_indenter)
      
    else:
      self.mesh = IndentationMesh(
        Na = self.mesh_Na, 
        Nb = self.mesh_Nb, 
        Ns = self.mesh_Ns, 
        Nf = self.mesh_Nf, 
        l = mesh_l)
      self.indenter = DeformableCone2D(
        half_angle = self.indenter_half_angle, 
        l = mesh_l, 
        Na = self.mesh_Na * (self.indenter_mesh_Na == 0) + self.indenter_mesh_Na * (self.indenter_mesh_Na != 0), 
        Nb = self.mesh_Nb * (self.indenter_mesh_Nb == 0) + self.indenter_mesh_Nb * (self.indenter_mesh_Nb != 0), 
        Ns = self.mesh_Ns * (self.indenter_mesh_Ns == 0) + self.indenter_mesh_Ns * (self.indenter_mesh_Ns != 0), 
        Nf = self.mesh_Nf * (self.indenter_mesh_Nf == 0) + self.indenter_mesh_Nf * (self.indenter_mesh_Nf != 0), 
        rigid = self.rigid_indenter)
    self.steps = [                                              
      Step(name='loading0', 
        nframes = self.frames, 
        disp = self.max_disp/2., 
        boundaries_3D= self.three_dimensional),
      Step(name='loading1', 
        nframes = self.frames, 
        disp = self.max_disp,
        boundaries_3D= self.three_dimensional), 
      Step(name = 'unloading', 
        nframes = self.frames, 
        disp = 0.,
        boundaries_3D= self.three_dimensional)] 
    if self.sample_mat_type == 'hollomon':
      self.sample_mat = Hollomon(
        labels = 'SAMPLE_MAT', 
        E =  self.sample_mat_args['young_modulus'], 
        nu = self.sample_mat_args['poisson_ratio'], 
        sy = self.sample_mat_args['yield_stress'],
        n = self.sample_mat_args['hardening'])
    if self.sample_mat_type == 'druckerprager':
      self.sample_mat = DruckerPrager(
        labels = 'SAMPLE_MAT', 
        E =  self.sample_mat_args['young_modulus'], 
        nu = self.sample_mat_args['poisson_ratio'], 
        sy = self.sample_mat_args['yield_stress'],
        beta = self.sample_mat_args['beta'],
        psi = self.sample_mat_args['psi'],
        k = self.sample_mat_args['k'])   
    if self.sample_mat_type == 'vonmises':
      self.sample_mat = VonMises(
        labels = 'SAMPLE_MAT', 
        E =  self.sample_mat_args['young_modulus'], 
        nu = self.sample_mat_args['poisson_ratio'], 
        sy = self.sample_mat_args['yield_stress'])   
    if self.sample_mat_type == 'elastic':
      self.sample_mat = Elastic(
        labels = 'SAMPLE_MAT', 
        E = self.sample_mat_args['young_modulus'], 
        nu = self.sample_mat_args['poisson_ratio'])  
    if self.indenter_mat_type == 'elastic':
      self.indenter_mat = Elastic(
        labels = 'INDENTER_MAT', 
        E = self.indenter_mat_args['young_modulus'], 
        nu = self.indenter_mat_args['poisson_ratio'])  
    
  
  
  def run(self, work_dir = 'workdir/', abqlauncher = '/opt/Abaqus/6.9/Commands/abaqus'):
    print '# Running {0}: id={1}, frames = {2}'.format(self.__class__.__name__, self.id, self.frames)
    self.preprocess()
    from abapy.indentation import Manager
    import numpy as np
    from copy import copy
    simname = self.__class__.__name__ + '_{0}'.format(self.id)
    # Creating abq postproc script
    f = open('{0}{1}_abqpostproc.py'.format(work_dir,self.__class__.__name__),'w')
    name = self.__class__.__name__ + '_' + str(self.id)
    out = self.abqpostproc_byRpt().replace('#FILE_NAME', name)
    f.write(out)
    f.close()
    abqpostproc = '{0}_abqpostproc.py'.format(self.__class__.__name__)
    #---------------------------------------
    # Setting simulation manager
    m = Manager()
    m.set_abqlauncher('/opt/Abaqus/6.9/Commands/abaqus')
    m.set_workdir(work_dir)
    m.set_simname(self.__class__.__name__ + '_{0}'.format(self.id))
    m.set_abqpostproc(abqpostproc)
    m.set_samplemesh(self.mesh)
    m.set_samplemat(self.sample_mat)
    m.set_indentermat(self.indenter_mat)
    m.set_friction(self.friction)
    m.set_steps(self.steps)
    m.set_indenter(self.indenter)
    m.set_is_3D(self.three_dimensional)
    m.set_pypostprocfunc(lambda data: data) # Here we just want to get back data
    #---------------------------------------
    # Running simulation and post processing
    m.erase_files()           # Workdir cleaning
    m.make_inp()              # INP creation
    m.run_sim()               # Running the simulation
    m.run_abqpostproc()       # First round of post processing in Abaqus
    data = m.run_pypostproc() # Second round of custom post processing in regular Python
    #m.erase_files()          # Workdir cleaning
    #---------------------------------------
    if data['completed']:
      print '# Simulation completed.'
      # Storing raw data
      self.completed = True
      sweep_factor = 1. # This factor aims to modify values that are affected by the fact that only a portion of the problem is solved due to symmetries.
      if self.three_dimensional: sweep_factor = 360. / self.sweep_angle
      self.force_hist = - sweep_factor * data['history']['force'] 
      self.disp_hist = -data['history']['disp'] 
      self.elastic_work_hist = sweep_factor * data['history']['allse'] 
      self.plastic_work_hist = sweep_factor * data['history']['allpd'] 
      self.friction_work_hist = sweep_factor * data['history']['allfd'] 
      self.total_work_hist = sweep_factor * data['history']['allwk'] 
      self.tip_penetration_hist = data['history']['tip_penetration'] 
      self.disp_field = data['field']['U']
      self.ind_disp_field = data['field']['Uind']
      self.stress_field = data['field']['S']
      self.total_strain_field = data['field']['LE']
      self.elastic_strain_field = data['field']['EE']
      self.plastic_strain_field = data['field']['PE']
      self.equivalent_plastic_strain_field = data['field']['PEEQ']
      self.contact_data = data['history']['contact']
      # Updating data
      self.Load_prefactor(update = True)
      self.Irreversible_work_ratio(update = True)
      self.Plastic_work_ratio(update = True)
      self.Tip_penetration(update = True)
      self.Contact_area(update = True)
      self.Unloading_fit()
    #---------------------------------------
    else:
      print '# Simulation not completed.'
#----------------------------------------------------------------------------------------------------------------      



#----------------------------------------------------------------------------------------------------------------    
# DATABASE MANAGEMENT CLASS
# 
class Database_Manager:
  def __init__(self, database_dir, database_name, cls , work_dir, abqlauncher):
    db = 'sqlite:///{0}{1}.db'.format(database_dir, database_name)
    engine = create_engine(db,echo=False)
    Base.metadata.create_all(engine) 
    Session = sessionmaker(bind=engine)
    self.session = Session()
    self.cls = cls
    self.work_dir = work_dir
    self.abqlauncher = abqlauncher
    
  def add_simulation(self, simulation):
    '''
    Adds a simulation to the database.
    
    Args:
    * simulation: Simulation class instance
    '''
    if isinstance(simulation, self.cls) == False: raise Exception, 'simulation must be Simulation instance'
    try:
      self.session.add(simulation)
      self.session.commit()
    except: 
      print 'simulation already exists or has not been declared corectly, nothing changed' 
      self.session.rollback()
      
  def get_next(self):
    '''
    Returns the next simulation to do regarding to difficulty criterias defined under.
    '''
    simus = self.session.query(self.cls).filter(self.cls.completed == False)
    if simus.all() != []:
      # finding max priority
      max_priority = self.session.query(self.cls).filter(self.cls.completed == False).order_by(desc(self.cls.priority)).first().priority
      # finding less difficult simulation with max priority
      simus = simus.filter(self.cls.priority == max_priority)
      simus = sorted(simus, key=lambda simu: simu.difficulty())
      simu = simus[0]
      # Adjusting number of requested frames
      diff = simu.difficulty()
      completed_simus = self.session.query(self.cls).filter(self.cls.completed == True)
      for csim in completed_simus:
        if csim.difficulty() <= diff:
          if csim.frames > simu.frames: simu.frames = csim.frames
      completed_simus = [c_simu for c_simu in completed_simus if c_simu.difficulty <= diff]
      self.session.commit()
    else:
      simu = None
    return simu
    
  def run_next(self):
    '''
    Runs the next simulation.
    '''
    simu = self.get_next()
    if simu != None:
      while True:
        simu.run(work_dir = self.work_dir, abqlauncher = self.abqlauncher)  
        if simu.completed: break
        simu.frames = int(simu.frames * 1.5)
        self.session.commit()
        print '# Number of frames changed to {0}.'.format(simu.frames)
      self.session.commit()
    else:
      print '# No more simulations to run'
  
  def run_all(self):
    '''
    Runs all the simulation in the right order until they all have been completed.
    '''
    while True:
      left_sims = self.session.query(self.cls).filter(self.cls.completed == False).count()
      if left_sims == 0: 
        print '# All simulations have been run.'
        break
      print '# {0} simulations left to run.'.format(left_sims)
      self.run_next()
  
  def query(self):
    '''
    Shortcut for database queries.
    '''
    return self.session.query(self.cls)
 #----------------------------------------------------------------------------------------------------------------
