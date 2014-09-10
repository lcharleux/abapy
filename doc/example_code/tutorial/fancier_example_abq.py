# ABAQUS/PYTHON POST PROCESSING SCRIPT
# Run using abaqus python / abaqus viewer -noGUI / abaqus cae -noGUI

# Packages (Abaqus, Abapy and built-in only here)
from odbAccess import openOdb
from abapy.misc import dump
from abapy.postproc import GetHistoryOutputByKey as gho
from abaqusConstants import JOB_STATUS_COMPLETED_SUCCESSFULLY


# Setting up some pathes
workdir = 'workdir'
name = 'indentation_axi_fancier'

# Opening the Odb File
odb = openOdb(workdir + '/' + name + '.odb')

# Testing job status
data = {}
status = odb.diagnosticData.jobStatus
if status == JOB_STATUS_COMPLETED_SUCCESSFULLY:
  data['completed'] = True
  # Finding back the position of the reference node of the indenter. Its number is stored inside a node set named REF_NODE.
  ref_node_label = odb.rootAssembly.instances['I_INDENTER'].nodeSets['REF_NODE'].nodes[0].label

  # Getting back the reaction forces along Y (RF2) and displacements along Y (U2) where they are recorded.
  RF2 = gho(odb, 'RF2')
  U2  = gho(odb, 'U2')

  # Packing data
  data['ref_node_label'] = ref_node_label
  data['RF2'] = RF2 
  data['U2']  = U2

  
else:
  data['completed'] = False
# Dumping data
dump(data, workdir + '/' + name + '.pckl')
# Closing Odb
odb.close()

