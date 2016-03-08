from abapy.postproc import ReadFieldOutputReport
import numpy as np
import time

def dummyReport(path, nl = 1000, nv = 10000):
  header = """********************************************************************************
Field Output Report, written Fri Apr 27 10:07:54 2012

Source 1
---------

   ODB: /home/lcharleux/Documents/Prog/modules/python/abapy/doc/example_code/postproc/indentation.odb
   Step: LOADING0
   Frame: Increment     10: Step Time =    1.000

Loc 1 : Nodal values from source 1

Output sorted by column "Node Label".

Field Output reported at nodes for part: I_SAMPLE

            Node     U.Magnitude
           Label          @Loc 1
---------------------------------\n"""
  labels = np.random.randint(1,nl+1, size = nv)
  values = np.random.rand(nv)
  out = header
  for i in xrange(nv): out += "{0} {1} \n".format(labels[i], values[i])
  open(path, "w").write(out)
    
path = 'test.rpt'
nv = 100000
dummyReport(path, nv = nv)

t0 = time.time()
field = ReadFieldOutputReport(path, position = 'nodes', dti = 'I', dtf = 'f')
t1 = time.time()

print "Read {0} values in {1} s".format(nv, t1-t0)


