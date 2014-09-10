from abapy.postproc import FieldOutput, TensorFieldOutput, VectorFieldOutput, Identity_like
data11 = [0., 0., 1.]
data22 = [0., 0., -1]
data12 = [1., 2., 0.]
labels = range(1,len(data11)+1)
fo11 = FieldOutput(labels = labels, data=data11,position='node')
fo22 = FieldOutput(labels = labels, data=data22,position='node')
fo12 = FieldOutput(labels = labels, data=data12,position='node')
tensor = TensorFieldOutput(data11 = fo11, data22 = fo22, data12 = fo12 )
t1, t2, t3, v1, v2, v3 = tensor.eigen()
