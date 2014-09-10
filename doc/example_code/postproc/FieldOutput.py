from abapy.postproc import FieldOutput
data = [-1.,5.,3.]
labels = [1,3,2]
fo = FieldOutput(data=data, labels = labels, position = 'node')
print fo # data is sorted by labels
print fo[1:2] # slicing
print fo[2] # indexing
print fo[1,3] # multiple indexing
print fo*2 # multiplication
fo2 = fo**2  #power
print fo2
print fo * fo2
print fo + fo2
print abs(fo) 

