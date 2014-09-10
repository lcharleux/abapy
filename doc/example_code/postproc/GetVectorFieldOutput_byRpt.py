from odbAccess import openOdb
from abapy.postproc import GetVectorFieldOutput_byRpt
odb_name = 'indentation.odb'
odb = openOdb(odb_name)
U = GetVectorFieldOutput_byRpt(
  odb = odb, 
  instance = 'I_SAMPLE', 
  step = 0,
  frame = -1,
  original_position = 'NODAL', 
  new_position = 'NODAL', 
  position = 'node',
  field = 'U', 
  sub_set_type = 'element', 
  sub_set = 'CORE',
  delete_report = True)


