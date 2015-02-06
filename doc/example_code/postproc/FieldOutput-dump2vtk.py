from abapy.postproc import FieldOutput
from abapy.mesh import Mesh, Nodes
x = [0.,1.,0.]
y = [0.,0.,1.]
z = [0.,0.,0.]
labels = [1,2,3]
nodes = Nodes(x=x,y=y,z=z, labels=labels)
mesh = Mesh(nodes=nodes)
mesh.add_element(label = 1 , connectivity = [1,2,3], space = 2 , name = 'tri3') # triangle element
nodeField = FieldOutput()
nodeField.add_data(data = 0., label = 1)
nodeField.add_data(data = 10., label = 2)
nodeField.add_data(data = 20., label = 3)
elementField = FieldOutput(position='element')
elementField.add_data(label = 1, data =10.)
out = ''
out+=mesh.dump2vtk()
out+=nodeField.dump2vtk('nodeField')
out+=elementField.dump2vtk('elementField')
f = open("FieldOutput-dump2vtk.vtk", "w")
f.write(out)
f.close()
