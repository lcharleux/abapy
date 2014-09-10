from abapy.postproc import FieldOutput, TensorFieldOutput, Identity_like
data1 = [1,2,3,5,6,]
data2 = [1. for i in data1]
labels = range(1,len(data1)+1)
fo1, fo2 = FieldOutput(labels = labels, data=data1, position='node' ), FieldOutput(labels = labels, data=data2,position='node')
tensor = TensorFieldOutput(data11 = fo1, data22 = fo2 )
identity = Identity_like(tensor)
