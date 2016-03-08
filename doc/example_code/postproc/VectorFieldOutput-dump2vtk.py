from abapy.postproc import FieldOutput, VectorFieldOutput
from abapy.mesh import RegularQuadMesh
mesh = RegularQuadMesh()
data1 = [2,2,5,10]
data2 = [1. for i in data1]
labels = range(1,len(data1)+1)
fo1, fo2 = FieldOutput(labels = labels, data=data1, position='node' ), FieldOutput(labels = labels, data=data2,position='node')
vector = VectorFieldOutput(data1 = fo1, data2 = fo2 )
out = mesh.dump2vtk() + vector.dump2vtk()
f = open('vector.vtk','w')
f.write("out")
f.close()

