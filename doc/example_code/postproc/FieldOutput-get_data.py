from abapy.postproc import FieldOutput
data = [1,2,3,5,6,0]
labels = range(1,len(data)+1)
fo = FieldOutput(data=data, labels = labels)
print fo.get_data(6)
print fo.get_data(10)
