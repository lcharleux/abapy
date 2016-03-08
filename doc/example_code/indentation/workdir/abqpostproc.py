# ABQPOSTPROC.PY
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
file_name = 'indentation'
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
  
  fo['Sind'] = [
    gtfo(odb = odb, 
      instance = 'I_INDENTER', 
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
dump(data, file_name+'.pckl')
