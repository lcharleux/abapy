from abapy.postproc import FieldOutput, VectorFieldOutput
data1 = [1,2,3,5,6,0]
data2 = [1. for i in data1]
labels = range(1,len(data1)+1)
fo1, fo2 = FieldOutput(data=data1, labels = labels), FieldOutput(data=data2, labels = labels)
vector = VectorFieldOutput(data1 = fo1, data2 = fo2 )
print vector.get_data(6)
x, y, z = vector.get_data(5)
print x, y, z
print vector.get_data(10)
norm = vector.get_norm()

