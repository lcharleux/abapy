from abapy.postproc import GetFieldOutput
from odbAccess import openOdb
odb = openOdb('indentation.odb')
U2 = GetFieldOutput(odb, step = 'LOADING0', frame = -1, instance ='I_SAMPLE', position =  'node', field = 'U', subField = 'U1') # Gets U2 at all nodes of instance 'I_SAMPLE'
U1 = GetFieldOutput(odb, step = 'LOADING0', frame = -1, instance ='I_SAMPLE', position =  'node', field = 'U', subField = 'U1', labels = [5,6]) # Here labels refer to nodes 5 and 6
S11 = GetFieldOutput(odb, step = 'LOADING0', frame = -1, instance ='I_SAMPLE', position =  'node', field = 'S', subField = 'S11', labels = 'CORE') # Here labels refers to nodes belonging to the node set 'CORE'
S12 = GetFieldOutput(odb, step = 'LOADING0', frame = -1, instance ='I_SAMPLE', position =  'element', field = 'S', subField = 'S12', labels = 'CORE') # Here labels refers to nodes belonging to the node set 'CORE'
