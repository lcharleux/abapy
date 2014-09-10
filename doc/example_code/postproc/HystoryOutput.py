from abapy.postproc import HistoryOutput
time = [ [1., 2.,3.] , [3.,4.,5.] , [5.,6.,7.] ] # Time separated in 3 steps
data = [ [2.,2.,2.] , [3.,3.,3.] , [4.,4.,4.] ] # Data separated in 3 steps
Data = HistoryOutput(time, data)
print Data 
# +, *, **, abs, neg act only on data, not on time
print Data + Data + 1. # addition
print Data * Data * 2. # multiplication
print ( Data / Data ) / 2. # division
print Data ** 2
print abs(Data)
print Data[1] # step 1
print Data[0:2]
print Data[0,2]

