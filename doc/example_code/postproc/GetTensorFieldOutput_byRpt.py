from odbAccess import openOdb
from abapy.postproc import GetTensorFieldOutput_byRpt
odb_name = 'indentation.odb'
odb = openOdb(odb_name)
S = GetTensorFieldOutput_byRpt(
  odb = odb, 
  instance = 'I_SAMPLE', 
  step = 0,
  frame = -1,
  original_position = 'INTEGRATION_POINT', 
  new_position = 'NODAL', 
  position = 'node',
  field = 'S', 
  sub_set_type = 'element', 
  sub_set = 'CORE',
  delete_report = True)


