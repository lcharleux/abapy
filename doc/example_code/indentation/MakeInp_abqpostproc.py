# Launch with 'abaqus viewer -noGUI MakeInp_abqpostproc.py'
from odbAccess import openOdb
from abapy.postproc import GetTensorFieldOutput_byRpt, GetMesh
from abapy.misc import dump
path_to_odb = 'workdir/'
odb_name = path_to_odb + 'indentation_axi.odb'
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
  delete_report = True)
mesh = GetMesh(odb = odb, instance = 'I_SAMPLE' )
dump(S, path_to_odb +  'indentation_S.pckl')
dump(mesh, path_to_odb + 'indentation_mesh.pckl')
