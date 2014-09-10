from abapy.postproc import GetMesh  
from odbAccess import openOdb
odb = openOdb('myOdb.odb')      
mesh = GetMesh(odb,'MYINSTANCE')
