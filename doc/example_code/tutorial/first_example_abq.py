# ABAQUS/PYTHON POST PROCESSING SCRIPT
# Run using abaqus python / abaqus viewer -noGUI / abaqus cae -noGUI

# Packages (Abaqus, Abapy and built-in only here)
from odbAccess import openOdb
from abapy.misc import dump
from abapy.postproc import GetHistoryOutputByKey as gho

# Setting up some pathes
workdir = 'workdir'
name = 'indentation_axi'

# Opening the Odb File
odb = openOdb(workdir + '/' + name + '.odb')

# Finding back the position of the reference node of the indenter. Its number is stored inside a node set named REF_NODE.
ref_node_label = odb.rootAssembly.instances['I_INDENTER'].nodeSets['REF_NODE'].nodes[0].label

# Getting back the reaction forces along Y (RF2) and displacements along Y (U2) where they are recorded.
RF2 = gho(odb, 'RF2')
U2  = gho(odb, 'U2')

# Packing data
data = {'ref_node_label': ref_node_label, 'RF2':RF2, 'U2':U2}

# Dumping data
dump(data, workdir + '/' + name + '.pckl')

# Closing Odb
odb.close()

