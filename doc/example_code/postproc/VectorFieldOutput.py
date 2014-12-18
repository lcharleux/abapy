from abapy.postproc import FieldOutput, VectorFieldOutput
data1 = [1,2,3,5,6,0]
data2 = [1. for i in data1]
labels = range(1,len(data1)+1)
fo1, fo2 = FieldOutput(labels = labels, data=data1, position='node' ), FieldOutput(labels = labels, data=data2,position='node')
vector = VectorFieldOutput(data1 = fo1, data2 = fo2 )
vector2 = VectorFieldOutput(data2 = fo2 )
vector # short description
print vector # long description
print vector[6] # Returns a VectorFieldOutput instance
print vector[1,4,6] # Picking label by label
print vector[1:6:2] # Slicing
vector.get_data(6) # Returns 3 floats
vector.norm() # Returns norm
vector.sum() # returns the sum of coords
vector * vector2 # Itemwise product (like numpy, unlike matlab)
vector.dot(vector2) # Dot/Scalar product
vector.cross(vector2) # Cross/Vector product
vector + 2 # Itemwise addition
vector * 2 # Itemwise multiplication
vector / 2 # Itemwise division
vector / vector2 # Itemwise division between vectors (numpy way)
abs(vector) # Absolute value
vector ** 2 # Power
vector ** vector # Itemwize power
 


