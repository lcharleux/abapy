from abapy.postproc import GetFieldOutput, GetVectorFieldOutput, GetTensorFieldOutput
from odbAccess import openOdb
odb = openOdb('indentation.odb')
U = GetVectorFieldOutput(odb, step = 'LOADING', frame = -1, instance ='I_SAMPLE', position =  'node', field = 'U') 
