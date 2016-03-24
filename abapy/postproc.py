'''
Post Processing
===============
'''




def GetMesh(odb,instance,dti='I'):
  '''Retrieves mesh on an instance in an Abaqus Output Database. 
  
  :param odb: output database
  :type odb: odb object
  :param instance: instance name declared in the Abaqus inp file.
  :type instance: string
  :param dti: int data type in array.array
  :type dti: 'I' or 'H'
  
  :rtype: Mesh instance
  
  .. literalinclude:: example_code/postproc/GetMesh.py
  
  '''
  from mesh import Mesh, Nodes
  from array import array
  inst = odb.rootAssembly.instances[instance]
  # Managing precision: float32 or float64
  precision = str(odb.jobData.precision)
  if precision == 'SINGLE_PRECISION': dtf = 'f'
  if precision == 'DOUBLE_PRECISION': dtf = 'd'
  nodes = Nodes(dtf=dtf,dti=dti)
  mesh = Mesh(nodes=nodes)
  ni = nodes.labels.index
  # Managing nodes
  for node in inst.nodes:
    c = node.coordinates
    l = node.label
    nodes.add_node(l,c[0],c[1],c[2])
  for nsk in inst.nodeSets.keys():
    nset = inst.nodeSets[nsk].nodes
    ns = [node.label for node in nset]
    nodes.add_set(nsk,ns)
  # Managing elements
  embeddedSpace = str(inst.embeddedSpace)
  if embeddedSpace in ['TWO_D_PLANAR', 'AXISYMMETRIC']: space =2 # To be completed 
  if embeddedSpace in ['THREE_D']: space =3 # To be completed 
  for element in inst.elements:
    l = element.label
    c = element.connectivity
    name = str(element.type)
    mesh.add_element(label = l, connectivity = c ,space = space , name = name)
  for esk in inst.elementSets.keys():
    eset = inst.elementSets[esk].elements
    es = [element.label for element in eset]
    mesh.add_set(esk,es)
  return mesh







class HistoryOutput(object):
  ''' Stores history output data from and allows useful operations. The key idea of this class is to allow easy storage of time dependant data without merging steps to allow further separating of each test steps (loading, unloading,...). The class allows additions, multiplication, ... between class instances and between class instances and int/floats. These operations only affect y data as long as time has no reason to be affected. 
  
  :param time: time represented by nested lists, each one corresponding to a step.
  :type time: list of list/array.array containing floats
  :param data: data (ex: displacement, force, energy ...).  It is represented by nested lists, each one corresponding to a step.
  :type data: list of list/array.array containing floats
  :param dtf: float data type used by array.array
  :type dtf: 'f', 'd'
  

  >>> from abapy.postproc import HistoryOutput
  >>> time = [ [1., 2.,3.] , [3.,4.,5.] , [5.,6.,7.] ] # Time separated in 3 steps
  >>> data = [ [2.,2.,2.] , [3.,3.,3.] , [4.,4.,4.] ] # Data separated in 3 steps
  >>> Data = HistoryOutput(time, data)
  >>> print Data 
  Field output instance: 3 steps
  Step 0: 3 points
  Time	Data
  1.0	2.0
  2.0	2.0
  3.0	2.0
  Step 1: 3 points
  Time	Data
  3.0	3.0
  4.0	3.0
  5.0	3.0
  Step 2: 3 points
  Time	Data
  5.0	4.0
  6.0	4.0
  7.0	4.0
  >>> # +, *, **, abs, neg act only on data, not on time
  ... print Data + Data + 1. # addition
  Field output instance: 3 steps
  Step 0: 3 points
  Time	Data
  1.0	5.0
  2.0	5.0
  3.0	5.0
  Step 1: 3 points
  Time	Data
  3.0	7.0
  4.0	7.0
  5.0	7.0
  Step 2: 3 points
  Time	Data
  5.0	9.0
  6.0	9.0
  7.0	9.0
  >>> print Data * Data * 2. # multiplication
  Field output instance: 3 steps
  Step 0: 3 points
  Time	Data
  1.0	8.0
  2.0	8.0
  3.0	8.0
  Step 1: 3 points
  Time	Data
  3.0	18.0
  4.0	18.0
  5.0	18.0
  Step 2: 3 points
  Time	Data
  5.0	32.0
  6.0	32.0
  7.0	32.0
  >>> print ( Data / Data ) / 2. # division
  Field output instance: 3 steps
  Step 0: 3 points
  Time	Data
  1.0	0.5
  2.0	0.5
  3.0	0.5
  Step 1: 3 points
  Time	Data
  3.0	0.5
  4.0	0.5
  5.0	0.5
  Step 2: 3 points
  Time	Data
  5.0	0.5
  6.0	0.5
  7.0	0.5
  >>> print Data ** 2
  Field output instance: 3 steps
  Step 0: 3 points
  Time	Data
  1.0	4.0
  2.0	4.0
  3.0	4.0
  Step 1: 3 points
  Time	Data
  3.0	9.0
  4.0	9.0
  5.0	9.0
  Step 2: 3 points
  Time	Data
  5.0	16.0
  6.0	16.0
  7.0	16.0
  >>> print abs(Data)
  Field output instance: 3 steps
  Step 0: 3 points
  Time	Data
  1.0	2.0
  2.0	2.0
  3.0	2.0
  Step 1: 3 points
  Time	Data
  3.0	3.0
  4.0	3.0
  5.0	3.0
  Step 2: 3 points
  Time	Data
  5.0	4.0
  6.0	4.0
  7.0	4.0
  >>> print Data[1] # step 1
  Field output instance: 1 steps
  Step 0: 3 points
  Time	Data
  3.0	3.0
  4.0	3.0
  5.0	3.0
  >>> print Data[0:2]
  Field output instance: 2 steps
  Step 0: 3 points
  Time	Data
  1.0	2.0
  2.0	2.0
  3.0	2.0
  Step 1: 3 points
  Time	Data
  3.0	3.0
  4.0	3.0
  5.0	3.0
  >>> print Data[0,2]
  Field output instance: 2 steps
  Step 0: 3 points
  Time	Data
  1.0	2.0
  2.0	2.0
  3.0	2.0
  Step 1: 3 points
  Time	Data
  5.0	4.0
  6.0	4.0
  7.0	4.0'''
  
  def __init__(self,time = [], data = [], dtf='f'):
    from array import array
    time_size = [len(t) for t in time]
    data_size = [len(d) for d in data]
    if data_size != time_size: raise Exception, 'time and data must have the same structure.'
    self.time = []
    self.data = []
    self.dtf = dtf
    for i in xrange(len(time)): 
      time_step = time[i]
      data_step = data[i]
      self.add_step(time_step, data_step)
  def add_step(self,time_step,data_step):
    '''
    Adds data to an HistoryOutput instance.
    
    :param time_step: time data to be added.
    :type time_step: list, array.array, np.array containing floats
    :param data_step: data to be added.
    :type data_step: list, array.array, np.array containing floats
    
    >>> from abapy.postproc import HistoryOutput
    >>> time = [ [0.,0.5, 1.] , [1., 1.5, 2.] ]
    >>> force = [ [4.,2., 1.] , [1., .5, .2] ] ]
    >>> Force = HistoryOutput(time,force)
    >>> Force.time # time
    [array('f', [0.0, 0.5, 1.0]), array('f', [1.0, 1.5, 2.0])]
    >>> Force.add_step([5.,5.,5.],[4.,4.,4.]) 
    >>> Force.time
    [array('f', [0.0, 0.5, 1.0]), array('f', [1.0, 1.5, 2.0]), array('f', [5.0, 5.0, 5.0])]
    >>> Force.data
    [array('f', [4.0, 2.0, 1.0]), array('f', [1.0, 0.5, 0.20000000298023224]), array('f', [4.0, 4.0, 4.0])]
    '''
    from array import array
    dtf = self.dtf
    zipped = zip(time_step, data_step) # fancy sorting using zip
    time_step, data_step = zip( *sorted(zipped) )
    self.time.append(array(dtf,time_step))
    self.data.append(array(dtf,data_step))
    
  def plotable(self):
    '''
    Gives back plotable version of the history output. Plotable differs from toArray on one point, toArray will concatenate steps into one single array for x and one for y where plotable will add None between steps before concatenation. Adding None allows matplotlib to draw discontinuous lines between steps without requiring ploting several independant arrays. By the way, the None methode is far faster.
    
    
    :rtype: 2 lists of floats and None
      
    .. plot:: example_code/postproc/historyOutput-plotable.py
       :include-source:

    '''
    time0, data0 = [],[]
    for i in xrange(len(self.time)):
      time0 += self.time[i].tolist()+[None]
      data0 += self.data[i].tolist()+[None]
    return time0, data0
    
  def toArray(self):
    '''
    Returns an array.array of concatenated steps for x and y.
    
    :rtype: array.array 
    
    >>> from abapy.postproc import HistoryOutput
    >>> time = [ [1., 2.,3.] , [3.,4.,5.] , [5.,6.,7.] ]
    >>> force = [ [2.,2.,2.] , [3.,3.,3.] , [4.,4.,4.] ]
    >>> Force = HistoryOutput(time, force)
    >>> x,y = Force.toArray()
    >>> x
    array('f', [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0])
    >>> y
    array('f', [2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0])


    '''
    from array import array
    dtf = self.dtf
    time0, data0 = array(dtf,[]),array(dtf,[])
    for i in xrange(len(self.time)):
      time0 += self.time[i]
      data0 += self.data[i]
    return time0, data0
  
  def _PreProcess(self, other):
    from numpy import array as n_array
    from numpy import float32, float64, ones_like
    if type(other) == type(self): # Other is one of us !
      if self.dtf == 'd':
        dtf = 'd'
      if other.dtf == 'd': 
        dtf = 'd'
        time = other.time
      else:
        dtf = 'f'
        time = self.time
      if dtf == 'f': n_dtf = float32
      if dtf == 'd': n_dtf = float64
      n_other = [n_array(o, dtype = n_dtf) for o in other.data ]
      n_self = [n_array(s, dtype = n_dtf) for s in self.data ]
    if type(other) in [float, long, int]:
      other = float(other)
      dtf = self.dtf
      if dtf == 'f': n_dtf = float32
      if dtf == 'd': n_dtf = float64
      n_self = [n_array(s, dtype = n_dtf) for s in self.data ]
      n_other = [other * ones_like(s) for s in n_self ]
      time = self.time
    return time, n_self, n_other, dtf
    
  def _PostProcess(self, time, n_out, dtf):
    from array import array
    a_out = [array(dtf, no.tolist()) for no in n_out]
    out = HistoryOutput(time = time, data = a_out, dtf = dtf)  
    return out  
      
  def __add__(self,other):
    time, n_self, n_other, dtf = self._PreProcess(other)
    n_out = [ n_self[i] + n_other[i] for i in xrange(len(n_self)) ] 
    out = self._PostProcess(time, n_out, dtf)
    return out
    
  __radd__ = __add__
  
  def __mul__(self,other):
    time, n_self, n_other, dtf = self._PreProcess(other)
    n_out = [ n_self[i] * n_other[i] for i in xrange(len(n_self)) ] 
    out = self._PostProcess(time, n_out, dtf)
    return out
   
  __rmul__ = __mul__  
  
  def __sub__(self, other):
    return self+(other*-1)
    
  def __rsub__(self, other):
    return (self*-1) + other
  
  def __div__(self,other):
    time, n_self, n_other, dtf = self._PreProcess(other)
    n_out = [ n_self[i] / n_other[i] for i in xrange(len(n_self)) ] 
    out = self._PostProcess(time, n_out, dtf)
    return out
  
  def __rdiv__(self,out):
    return self**-1 * out
  
  def __pow__(self,other):
    time, n_self, n_other, dtf = self._PreProcess(other)
    n_out = [ n_self[i] ** n_other[i] for i in xrange(len(n_self)) ] 
    out = self._PostProcess(time, n_out, dtf)
    return out
  
  def __neg__(self):
    return self*-1
    
  def __abs__(self):
    time, n_self, n_other, dtf = self._PreProcess(1)
    n_out = [ abs(n_self[i]) for i in xrange(len(n_self)) ] 
    out = self._PostProcess(time, n_out, dtf)
    return out
    
  def __getitem__(self,s):
    from array import array
    from copy import deepcopy
    labs = []
    if type(s) in [int, long]:
      labs = [s]
    if type(s) is slice:
      start = s.start
      stop  = s.stop
      step  = s.step
      if step == None: step = 1
      labs = range(start,stop,step)
    if type(s) in [tuple,list, array]:  
      for a in s:
        if type(a) in [int, long]:labs.append(a)
    dtf = self.dtf
    time = self.time
    data = self.data
    out = HistoryOutput(dtf = dtf)
    for i in labs: out.add_step(time[i], data[i])
    return out
    
  def __repr__(self):
    return '<HistoryOutput instance: {0} steps>'.format(len(self.time))
  
  def __str__(self):
    time, data = self.time, self.data
    nstep = len(time)
    out = 'History output instance: {0} steps\n'.format(nstep)
    pattern0 = 'Step {0}: {1} points\nTime\tData\n'
    pattern1 = '{0}\t{1}\n'
    for s in xrange(nstep):
      time_step, data_step = time[s], data[s]
      npoints = len(time_step)
      out += pattern0.format(s, npoints)
      for p in xrange(npoints):
        out += pattern1.format(time_step[p], data_step[p])
    return out    
  
  def total(self):
    '''
    Returns the total of all data.
    
    :rtype: float
    '''
    time, data = self.toArray()
    return sum(data)
    
  
  def average(self, method = 'trapz'):
    '''
    Returns the average of all data over time using ``integral``. This average is performed step by step to avoid errors due to disconnected steps.
     
    :param method: choice between trapezoid rule ('trapz') or Simpson rule ('simps').
    :type method: string
    :rtype: float
    
    >>> from abapy.postproc import HistoryOutput
    >>> from math import sin, pi
    >>> N = 100
    >>> hist = HistoryOutput()
    >>> time = [pi / 2 * float(i)/N for i in xrange(N+1)]
    >>> data = [sin(t) for t in time]
    >>> hist.add_step(time_step = time, data_step = data)
    >>> time2 = [10., 11.]
    >>> data2 = [1., 1.]
    >>> hist.add_step(time_step = time2, data_step = data2)
    >>> sol = 2. / pi + 1.
    >>> print 'Print computed value:', hist.average()
    Print computed value: 1.63660673935
    >>> print 'Analytic solution:', sol
    Analytic solution: 1.63661977237
    >>> print 'Relative error: {0:.4}%'.format( (hist.average() - sol)/sol * 100.)
    Relative error: -0.0007963%

    '''
    out, dt = 0., 0.
    for i in xrange(len(self.time)):
      dt += self[i].duration()
      out +=  self[i].integral(method = method)
    return out/dt
  
  def duration(self):
    '''
    Returns the duration of the output by computing max(time) - min(time).
    
    :rtype: float
    '''
    time, data = self.toArray()
    return max(time)-min(time)
    
  def data_min(self):
    '''
    Returns the minimum value of data.
    
    :rtype: float
    '''
    time, data = self.toArray()
    return min(data)
  
  def data_max(self):
    '''
    Returns the maximum value of data.
    
    :rtype: float
    '''
    time, data = self.toArray()
    return max(data)
   
  def time_min(self):
    '''
    Returns the minimum value of time.
    
    :rtype: float
    '''
    time, data = self.toArray()
    return min(time)  
    
  def time_max(self):
    '''
    Returns the maximum value of time.
    
    :rtype: float
    '''
    time, data = self.toArray()
    return max(time)
  
  def integral(self,method = 'trapz'):
    '''
    Returns the integral of the history output using the trapezoid or Simpson rule.
    
    :param method: choice between trapezoid rule ('trapz') or Simpson rule ('simps').
    :type method: string
    :rtype: float
    
    >>> from abapy.postproc import HistoryOutput
    >>> time = [ [0., 1.], [3., 4.] ]
    >>> data = [ [.5, 1.5], [.5, 1.5] ]
    >>> hist = HistoryOutput(time = time, data = data)
    >>> hist[0].integral()
    1.0
    >>> hist[1].integral()
    1.0
    >>> hist.integral()
    2.0
    >>> N = 10
    >>> from math import sin, pi
    >>> time = [pi / 2 * float(i)/N for i in xrange(N+1)]
    >>> data = [sin(t) for t in time]
    >>> hist = HistoryOutput()
    >>> hist.add_step(time_step = time, data_step = data)
    >>> trap = hist.integral()
    >>> simp = hist.integral(method = 'simps')
    >>> trap_error = (trap -1.)
    >>> simp_error = (simp -1.)
    
    
    Relative errors:
      * Trapezoid rule: -0.21%
      * Simpson rule: 0.00033%
      
    .. note:: uses ``scipy``
    '''
    
    from numpy import array, float32, float64
    from scipy.integrate import simps, trapz
    dtf = self.dtf
    if dtf == 'f': ndtf = float32
    if dtf == 'd': ndtf = float64
    if method not in ['trapz', 'simps']:
      raise Exception, 'method must be trapz or simps'
    if method == 'trapz': rule = trapz
    if method == 'simps': rule = simps
    time, data = self.time, self.data
    out = 0.
    for i in xrange(len(time)):
      t = array(time[i], dtype = ndtf)
      d = array(data[i], dtype = ndtf)
      out += rule(x=t, y=d)
      '''
      dt = t[1:] - t[:-1]
      d2 = (d[1:] + d[:-1])/2
      out += (d2 / dt).sum()
      '''
    return out  
    

     



def GetHistoryOutputByKey(odb,key):
  '''
  Retrieves an history output in an odb object using key (U2, EVOL, RF2,...)
  
  :param odb: Abaqus output database object produced by odbAcces.openOdb.
  :type odb: odb object
  :param key: name of the requested variable (*i. e.* 'U2', 'COOR1', ...)
  :type key: string
  :rtype: dict of HistoryOutput instance where keys are HistoryRegions names (*i. e.* locations)
  
  >>> from odbAccess import openOdb
  >>> odb = openOdb('mySimulation.odb')
  >>> from abapy.postproc import GetHistoryOutputByKey
  >>> u_2 = GetHistoryOutputByKey(odb,'U2')

  '''
  from array import array
  precision = str(odb.jobData.precision)
  if precision == 'SINGLE_PRECISION': dtf = 'f'
  if precision == 'DOUBLE_PRECISION': dtf = 'd'
  out = {}
  steps = odb.steps
  for stepk in steps.keys():
    hrs = steps[stepk].historyRegions
    startTime = steps[stepk].totalTime
    for hrk in hrs.keys():
      hos = hrs[hrk].historyOutputs
      hoks = hos.keys()
      if key in hoks:
        if hrk not in out.keys(): out[hrk] = HistoryOutput()
        output = hos[key].data
        time,data = array(dtf,[]), array(dtf,[]) 
        for a in output:
          time.append(a[0]+startTime)
          data.append(a[1])
        out[hrk].add_step(time_step = time, data_step = data)
  return out

 

   
def GetFieldOutput(odb, step, frame, instance, position, field, subField=None, labels=None,dti='I'):
  '''
  Retrieves a field output in an Abaqus odb object and stores it in a FieldOutput class instance. Field output that are classically available at integration points must be interpolated at nodes. This can be requested in the Abaqus *inp* file using: *Element Output, position = nodes*.
  
  
  
  
  
  
  :param odb: odb object produced by odbAccess.openOdb in abaqus python or abaqus viewer -noGUI
  :type odb: odb object.
  :param step: step name defined in the abaqus inp file. May be the upper case version of original string name.
  :type step: string
  :param frame: requested frame indice in the odb.
  :type frame: int
  :param instance: instance name defined in the abaqus odb file. May be the upper case version of the original name. 
  :type instance: string 
  :param position: position at which the output is to be computed.
  :type position: 'node', 'element'
  :param field: requested field output ('LE','S','U','AC YIELD',...).
  :type field: string
  :param subField: requested subfield in the case of non scalar fields, can be a component (U1, S12) or an invariant (mises, tresca, inv3, maxPrincipal). In the case of scalar fields, it has to be None
  :type subField: string or None
  :param labels: if not None, it provides a set of locations (elements/nodes labels or node/element set label) where the field is to be computed. if None, every available location is used and labels are sorted
  :type labels: list, array.array, numpy.array of unsigned non zero ints or string
  :param dti: int data type in ``array.array``
  :type dti: 'I', 'H'
  :rtype: ``FieldOutput`` instance
  
  .. note:: This function can only be executed in abaqus python or abaqus viewer -noGUI
  
  >>> from abapy.postproc import GetFieldOutput
  >>> from odbAccess import openOdb
  >>> odb = openOdb('indentation.odb')
  >>> U2 = GetFieldOutput(odb, step = 'LOADING0', frame = -1, instance ='I_SAMPLE', position =  'node', field = 'U', subField = 'U1') # Gets U2 at all nodes of instance 'I_SAMPLE'
  >>> U1 = GetFieldOutput(odb, step = 'LOADING0', frame = -1, instance ='I_SAMPLE', position =  'node', field = 'U', subField = 'U1', labels = [5,6]) # Here labels refer to nodes 5 and 6
  >>> S11 = GetFieldOutput(odb, step = 'LOADING0', frame = -1, instance ='I_SAMPLE', position =  'node', field = 'S', subField = 'S11', labels = 'CORE') # Here labels refers to nodes belonging to the node set 'CORE'
  >>> S12 = GetFieldOutput(odb, step = 'LOADING0', frame = -1, instance ='I_SAMPLE', position =  'element', field = 'S', subField = 'S12', labels = 'CORE') # Here labels refers to nodes belonging to the node set 'CORE'

  
  .. note::
       * If dti='H' is chosen, labels are stored as unsigned 16 bits ints. If more than 65k labels are stored, an OverFlow error will be raised.
       * This function had memory usage problems in its early version, these have been solved by using more widely array.array. It is still a bit slow but with the lack of numpy in Abaqus, no better solutions have been found yet. I'm open to any faster solution even involving the used temporary rpt files procuced by Abaqus'''
  
  from array import array
  from abaqusConstants import WHOLE_ELEMENT, NODAL, ELEMENT_NODAL, INTEGRATION_POINT, SCALAR
  Instance = odb.rootAssembly.instances[instance]
  # Finding precision
  precision = str(odb.jobData.precision)
  if precision == 'SINGLE_PRECISION': dtf = 'f'
  if precision == 'DOUBLE_PRECISION': dtf = 'd'
  # Building some references in the odb
  abqFieldOutput = odb.steps[step].frames[frame].fieldOutputs[field]
  if position == 'node': 
    getLabel = lambda x : x.nodeLabel
    abqPositions = [NODAL, ELEMENT_NODAL]
    positionLabel = 'nodeLabel'
    if labels == None: labels = [x.label for x in Instance.nodes]
    if type(labels) == str:
      abqNodeSet = odb.rootAssembly.instances[instance].nodeSets[labels].nodes
      labels = array(dti,[ abqNode.label for abqNode in abqNodeSet ])
  if position == 'element': 
    getLabel = lambda x : x.elementLabel
    abqPositions = [WHOLE_ELEMENT, INTEGRATION_POINT]
    if labels == None: labels = [x.label for x in Instance.elements]
    positionLabel = 'elementLabel'
    if type(labels) == str:
      abqElemSet = odb.rootAssembly.instances[instance].elementSets[labels].elements
      labels = array(dti,[ abqElem.label for abqElem in abqElemSet ])
  labels = array(dti,sorted(labels))
  li = labels.index
  Nitems  = len(labels) # Number of nodes/elements where the field is to be extracted
  availablePositions = [loc.position for loc in abqFieldOutput.locations] # Available Abaqus positions for the requested field
  for abqloc in availablePositions:
    if abqloc in abqPositions: matchedPosition = abqloc
  Values = abqFieldOutput.getSubset(region=Instance).getSubset(position = matchedPosition)
  Nvalues = len(Values.values)
  if subField == None:
    if Values.type == SCALAR:
      scalarValues = Values
    else:
      raise Exception, 'field output is not scalar, maybe because subfield is None.' 
  if subField != None: 
    fieldInvariants = abqFieldOutput.validInvariants # Warning: using getsubset or getscalarfield can lead abaqus to give wrong field invariants or field components. This bug bug seems to be avoided by requestions invariants and components before subseting or scalaring the field.
    fieldComponents = abqFieldOutput.componentLabels  
    for key in fieldComponents:
      if subField.lower() == key.lower(): scalarValues = Values.getScalarField(componentLabel = key)
    for key in fieldInvariants:
      if subField.lower() == str(key).lower():  scalarValues = Values.getScalarField(invariant = key)
  count = array(dti,[0 for i in xrange(Nitems)])
  temp =  array(dtf,[0. for i in xrange(Nitems)])
  for v in scalarValues.values:
    i = v.__getattribute__(positionLabel)
    if i in labels:
      l = li(i)
      count[l] += 1
      temp[l] += v.data
  out_data, out_labels = array(dtf,[]) , array(dti, [])  
  for i in xrange(Nitems):
    if count[i] != 0:
      out_data.append(temp[i]/count[i] )
      out_labels.append(labels[i])
    
  #data = array(dtf,[temp[i]/count[i] for i in xrange(Nitems)])
  out = FieldOutput(position = position, data =out_data, labels = out_labels, dtf = dtf )
  return out

def GetVectorFieldOutput(odb, step, frame, instance, position, field, labels=None,dti='I'):
  '''
  Returns a VectorFieldOutput from an odb object.
  
  :param odb: odb object produced by odbAccess.openOdb in abaqus python or abaqus viewer -noGUI
  :type odb: odb object.
  :param step: step name defined in the abaqus inp file. May be the upper case version of original string name.
  :type step: string
  :param frame: requested frame indice in the odb.
  :type frame: int
  :param instance: instance name defined in the abaqus odb file. May be the upper case version of the original name. 
  :type instance: string 
  :param position: position at which the output is to be computed.
  :type position: 'node', 'element'
  :param field: requested vector field output ('U',...).
  :type field: string
  :param labels: if not None, it provides a set of locations (elements/nodes labels or node/element set label) where the field is to be computed. if None, every available location is used and labels are sorted
  :type labels: list, array.array, numpy.array of unsigned non zero ints or string
  :param dti: int data type in ``array.array``
  :type dti: 'I', 'H'
  :rtype: ``VectorFieldOutput`` instance
  
  .. note:: This function can only be executed in abaqus python or abaqus viewer -noGUI
  
  >>> from abapy.postproc import GetFieldOutput, GetVectorFieldOutput
  >>> from odbAccess import openOdb
  >>> odb = openOdb('indentation.odb')
  >>> U = GetVectorFieldOutput(odb, step = 'LOADING', frame = -1, instance ='I_SAMPLE', position =  'node', field = 'U') 
  >>> odb.close()
  '''
  # First we need to find the number of components: if we are in 3D, it should be 3, in 2D it will be 2.
  components = odb.steps[step].frames[frame].fieldOutputs[field].componentLabels
  comp_number = len(components)
  # Now let's find the data we need
  Field = []
  for ncomp in xrange(comp_number):
    Field.append( GetFieldOutput(odb, step = step, frame = frame, instance = instance, position =  position , field = field, subField= components[ncomp], labels = labels, dti = dti )  )
  if len(Field) == 2:
    return VectorFieldOutput(position = position, data1 = Field[0],  data2 = Field[1])  
  if len(Field) == 3:
    return VectorFieldOutput(position = position, data1 = Field[0],  data2 = Field[1], data3 = Field[2])  


def GetTensorFieldOutput(odb, step, frame, instance, position, field, labels=None,dti='I'):
  '''
  Returns a TensorFieldOutput from an odb object.
  
  :param odb: odb object produced by odbAccess.openOdb in abaqus python or abaqus viewer -noGUI
  :type odb: odb object.
  :param step: step name defined in the abaqus inp file. May be the upper case version of original string name.
  :type step: string
  :param frame: requested frame indice in the odb.
  :type frame: int
  :param instance: instance name defined in the abaqus odb file. May be the upper case version of the original name. 
  :type instance: string 
  :param position: position at which the output is to be computed.
  :type position: 'node', 'element'
  :param field: requested tensor field output ('LE','S',...).
  :type field: string
  :param labels: if not None, it provides a set of locations (elements/nodes labels or node/element set label) where the field is to be computed. if None, every available location is used and labels are sorted
  :type labels: list, array.array, numpy.array of unsigned non zero ints or string
  :param dti: int data type in ``array.array``
  :type dti: 'I', 'H'
  :rtype: ``TensorFieldOutput`` instance
  
  .. note:: This function can only be executed in abaqus python or abaqus viewer -noGUI
  
  >>> from abapy.postproc import GetFieldOutput, GetVectorFieldOutput, GetTensorFieldOutput
  >>> from odbAccess import openOdb
  >>> odb = openOdb('indentation.odb')
  >>> S = GetTensorFieldOutput(odb, step = 'LOADING', frame = -1, instance ='I_SAMPLE', position =  'node', field = 'S')
  >>> odb.close()
  
  '''
  # First we need to find the number of components: if we are in 3D, it should be 6, in 2D it will be 4.
  components = odb.steps[step].frames[frame].fieldOutputs[field].componentLabels
  comp_number = len(components)
  # Now let's find the data we need
  Field = []
  for ncomp in xrange(comp_number):
    Field.append( GetFieldOutput(odb, step = step, frame = frame, instance = instance, position =  position , field = field, subField= components[ncomp] , labels = labels, dti = dti )  )
  if len(Field) == 4:
    return TensorFieldOutput(position = position, data11 = Field[0], data22 = Field[1], data33 = Field[2], data12 = Field[3])  
  if len(Field) == 6:
    return TensorFieldOutput(position = position, data11 = Field[0], data22 = Field[1], data33 = Field[2], data12 = Field[3], data13 = Field[4], data23 = Field[5])  


def MakeFieldOutputReport(odb, instance, step, frame, report_name, original_position, new_position, field, sub_field = None, sub_field_prefix = None, sub_set_type = None, sub_set = None):
  '''
  Writes a field output report using Abaqus. The major interrest of this function is that it is really fast compared to ``GetFieldOutput`` which tends to get badly slow on odbs containing more than 1500 elements. One other interrest is that it doesn't require to used ``position = nodes`` option in the INP file to evaluate fields at nodes. It is especially efficient when averaging is necessary (example: computing stress at nodes). The two drawbacks are that it requires ``abaqus viewer`` (or ``cae``) using the ``-noGUI`` where GetFieldOutput only requires ``abaqus python`` so it depends on the license server lag (which can be of several seconds). The second drawback is that it requires to write a file in place where you have write permission. This function is made to used in conjunction with ``ReadFieldOutputReport``. 
  
  :param odb: output database to be used.
  :type odb: odb instance produced by ``odbAccess.openOdb``
  :param instance: instance to use.
  :type instance: string
  :param step: step to use, this argument can be either the step number or the step label.
  :type step: string or int
  :param frame: frame number, can be negative for reverse counting.
  :type frame: int
  :param report_name: name or path+name of the report to written.
  :type report_name: string
  :param original_position: position at which the field is expressed. Can be 'NODAL', 'WHOLE_ELEMENT' or 'INTEGRATION_POINT'.
  :type string:
  :param new_position: position at which you would like the field to be expressed. Can be 'NODAL', 'WHOLE_ELEMENT' or 'INTEGRATION_POINT' or 'ELEMENT_NODAL'. Note that ``ReadFieldOutputReport`` will be capable of averaging values over elements when 'INTEGRATION_POINT' or 'ELEMENT_NODAL' option is sellected.
  :type new_position: string
  :param field: field to export, example: 'S', 'U', 'EVOL',...
  :type field: string
  :param sub_field: can be a component of an invariant, example: 11, 2, 'Mises', 'Magnitude'. Here the use of 'Mises' instead of 'MISES' can be surprising that's the way abaqus is written..
  :type sub_field: string or int
  :param sub_set: set to which the report is restricted, must be the label of an existing node or element set.
  :type sub_set: string
  :param sub_set_type: type of the sub_set, can be node or element.
  :type sub_set_type: string
  
  
  All examples below are performed on a small indentation ODB: 
  
  
  >>> from odbAccess import openOdb
  >>> from abapy.postproc import MakeFieldOutputReport
  >>> # Some settings
  >>> odb_name = 'indentation.odb'
  >>> report_name = 'indentation_core_step0_frame1_S11_nodes.rpt'
  >>> step = 0
  >>> frame = -1
  >>> new_position = 'NODAL'
  >>> original_position = 'INTEGRATION_POINT'
  >>> field = 'S'
  >>> sub_field = 11
  >>> instance = 'I_SAMPLE'
  >>> sub_set = 'CORE'
  >>> sub_set_type = 'element'
  >>> # Function testing
  >>> odb = openOdb(odb_name)
  >>> MakeFieldOutputReport(
  ...   odb = odb, 
  ...   instance = instance, 
  ...   step = step,
  ...   frame = frame,
  ...   report_name = report_name, 
  ...   original_position = original_position, 
  ...   new_position = new_position, 
  ...   field = field, 
  ...   sub_field = sub_field, 
  ...   sub_set_type = sub_set_type, 
  ...   sub_set = sub_set)
  >>> new_position = 'INTEGRATION_POINT'
  >>> report_name = 'indentation_core_step0_frame1_S11_elements.rpt'   
  >>> MakeFieldOutputReport(
  ...   odb = odb, 
  ...   instance = instance, 
  ...   step = step,
  ...   frame = frame,
  ...   report_name = report_name, 
  ...   original_position = original_position, 
  ...   new_position = new_position, 
  ...   field = field, 
  ...   sub_field = sub_field, 
  ...   sub_set_type = sub_set_type, 
  ...   sub_set = sub_set)
  >>> new_position = 'ELEMENT_NODAL'
  >>> report_name = 'indentation_core_step0_frame1_S11_element-nodal.rpt'   
  >>> MakeFieldOutputReport(
  ...   odb = odb, 
  ...   instance = instance, 
  ...   step = step,
  ...   frame = frame,
  ...   report_name = report_name, 
  ...   original_position = original_position, 
  ...   new_position = new_position, 
  ...   field = field, 
  ...   sub_field = sub_field, 
  ...   sub_set_type = sub_set_type, 
  ...   sub_set = sub_set)
  >>> field = 'U'
  >>> sub_field = 'Magnitude'
  >>> original_position = 'NODAL'
  >>> new_position = 'NODAL'
  >>> report_name = 'indentation_core_step0_frame1_U-MAG_nodal.rpt'   
  >>> MakeFieldOutputReport(
  ...   odb = odb, 
  ...   instance = instance, 
  ...   step = step,
  ...   frame = frame,
  ...   report_name = report_name, 
  ...   original_position = original_position, 
  ...   new_position = new_position, 
  ...   field = field, 
  ...   sub_field = sub_field, 
  ...   sub_set_type = sub_set_type, 
  ...   sub_set = sub_set)
  
  
  Four reports were produced:
    *  :download:`indentation_core_step0_frame1_S11_nodes.rpt <example_code/postproc/indentation_core_step0_frame1_S11_nodes.rpt>`
    *  :download:`indentation_core_step0_frame1_S11_elements.rpt <example_code/postproc/indentation_core_step0_frame1_S11_elements.rpt>`
    *  :download:`indentation_core_step0_frame1_S11_element-nodal.rpt <example_code/postproc/indentation_core_step0_frame1_S11_element-nodal.rpt>`
    *  :download:`indentation_core_step0_frame1_U-MAG_nodal.rpt <example_code/postproc/indentation_core_step0_frame1_U-MAG_nodal.rpt>`
  
  
  '''
  from abaqus import session, NumberFormat
  import visualization
  import displayGroupOdbToolset as dgo
  from abaqusConstants import NODAL, INTEGRATION_POINT, COMPONENT, INVARIANT, WHOLE_ELEMENT, ELEMENT_CENTROID, ELEMENT_NODAL, OFF, ENGINEERING
  #import pickle
  if original_position == 'INTEGRATION_POINT': original_abqposition = INTEGRATION_POINT
  if original_position == 'NODAL': original_abqposition = NODAL
  if original_position == 'WHOLE_ELEMENT': original_abqposition = WHOLE_ELEMENT
  if original_position == 'ELEMENT_NODAL': original_abqposition = ELEMENT_NODAL
  if new_position == 'NODAL':
    new_abqposition = NODAL
    sortItem = 'Node Label'
  if new_position == 'ELEMENT_NODAL':
    new_abqposition = ELEMENT_NODAL
    sortItem = 'Node Label'
  if new_position == 'WHOLE_ELEMENT':
    new_abqposition = WHOLE_ELEMENT
    sortItem = 'Element Label'
  if new_position == 'INTEGRATION_POINT':
    new_abqposition = INTEGRATION_POINT
    sortItem = 'Element Label'
  variable = [field]
  variable.append(original_abqposition)
  if type(step) == str: step = odb.steps.keys().index(step) 
  if sub_field != None:
    if sub_field_prefix == None:
      prefix = field
    else:
      prefix = sub_field_prefix  
    if type(sub_field) == int:
      sub_field_type = COMPONENT
      sub_field_flag = prefix+str(sub_field)
      print sub_field_flag 
    if type(sub_field) == str:
      sub_field_type = INVARIANT
      sub_field_flag = sub_field
    variable.append( ( (sub_field_type, sub_field_flag) , ) )  
  if sub_set == None:
    leaf = dgo.LeafFromPartInstance(instance)
  else:
    if sub_set_type == 'node':
      leaf = dgo.LeafFromNodeSets(instance + '.' + sub_set)
    if sub_set_type == 'element':
      leaf = dgo.LeafFromElementSets(instance + '.' + sub_set)
  print variable
  if frame < 0:
    frames_list = xrange(len(odb.steps[ odb.steps.keys()[step] ].frames))
    frame = frames_list[frame]
  session.viewports['Viewport: 1'].setValues(displayedObject=odb)
  dg = session.viewports['Viewport: 1'].odbDisplay.displayGroup
  dg = session.DisplayGroup(name='Dummy', leaf = leaf)
  session.viewports['Viewport: 1'].odbDisplay.setValues(visibleDisplayGroups=(dg, ))
  odb = session.odbs[odb.name]
  session.fieldReportOptions.setValues(
    printTotal = OFF, 
    printMinMax = OFF)
  nf = NumberFormat(numDigits=9, precision=0, format=ENGINEERING)
  session.writeFieldReport(
    fileName=report_name, 
    append=OFF, 
    sortItem=sortItem, 
    odb=odb, 
    step=step, 
    frame=frame, 
    outputPosition=new_abqposition, 
    variable=(variable,)
    )


""" # Deprecated due to low speed. See below for new version. LC
def ReadFieldOutputReport(report_name, position = 'node', dti = 'I', dtf = 'f'):
  '''
  Reads a report file generated by Abaqus (for example using ``MakeFieldOutputReport`` and converts it in FieldOutputFormat.
  
  :param report_name: report_name or path + name of the report to read.
  :type report_name: string
  :param position: position where the ``FieldOutput`` is to be declared. The function will look at the first and the last column of the report. The first will be considered as the label (*i. e.* element or node) and the last the value. In some case, like reports written using 'ELEMENT_NODAL' or 'INTEGRATION_POINT' locations, each label will appear several times. The present function will collect all the corresponding values and average them. At the end, the only possibilities for this parameter should be 'node' or 'element' as described in the doc of ``FieldOutput``.
  :type position: 'node' or 'element'
  :param dti: int data type in array.array
  :type dti: 'I', 'H'
  :param dtf: float data type in array.array
  :type dtf: 'f', 'd'
  :rtype: ``FieldOutput`` instance.
  
  .. note:: This function can be run either in abaqus python, abaqus viewer -noGUI, abaqus cae -noGUI and regular python.
  
  >>> from abapy.postproc import ReadFieldOutputReport
  >>> report_name = 'indentation_core_step0_frame1_S11_nodes.rpt'
  >>> S11 = ReadFieldOutputReport(report_name, position = 'nodes', dti = 'I', dtf = 'f')
  '''
  
  from abapy.postproc import FieldOutput
  from array import array
  f = open(report_name, 'r')
  lines = f.readlines() 
  f.close()
  labels = array( dti, [] )
  data = array( dtf , [] )
  counter = array( dti, [] )
  for i in xrange(len(lines)):
    line = lines[i]
    words = line.split()
    try:
      label = int(words[0])
      value = float(words[-1])
    except:
      pass  
    else:
      if label not in labels:
        labels.append(label)
        counter.append(1)
        data.append(value)
      else:
        pos = labels.index(label)
        counter[pos] += 1
        data[pos] += value
  for i in xrange(len(labels)):
    data[i] = data[i]/counter[i]
  return FieldOutput(labels = labels, data = data, position = position, dti = dti, dtf = dtf)      
"""

def ReadFieldOutputReport(report_name, position = 'node', dti = 'I', dtf = 'f'):
  '''
  Reads a report file generated by Abaqus (for example using ``MakeFieldOutputReport`` and converts it in FieldOutputFormat.
  
  :param report_name: report_name or path + name of the report to read.
  :type report_name: string
  :param position: position where the ``FieldOutput`` is to be declared. The function will look at the first and the last column of the report. The first will be considered as the label (*i. e.* element or node) and the last the value. In some case, like reports written using 'ELEMENT_NODAL' or 'INTEGRATION_POINT' locations, each label will appear several times. The present function will collect all the corresponding values and average them. At the end, the only possibilities for this parameter should be 'node' or 'element' as described in the doc of ``FieldOutput``.
  :type position: 'node' or 'element'
  :param dti: int data type in array.array
  :type dti: 'I', 'H'
  :param dtf: float data type in array.array
  :type dtf: 'f', 'd'
  :rtype: ``FieldOutput`` instance.
  
  .. note:: This function can be run either in abaqus python, abaqus viewer -noGUI, abaqus cae -noGUI and regular python.
  
  >>> from abapy.postproc import ReadFieldOutputReport
  >>> report_name = 'indentation_core_step0_frame1_S11_nodes.rpt'
  >>> S11 = ReadFieldOutputReport(report_name, position = 'nodes', dti = 'I', dtf = 'f')
  '''
  
  from abapy.postproc import FieldOutput
  from array import array
  f = open(report_name, 'r')
  lines = f.readlines() 
  f.close()
  slines = [line.split() for line in lines]
  # Getting labels
  labels  = []
  slines2 = []
  for words in slines:
    try:
      label = int(words[0])
      labels.append(label)
      slines2.append(words)
    except:
      pass
  labels = list(set(labels))
  counters = [0 for l in labels] 
  data = {}
  for l in labels: data[l] = [0., 0] # [value, counter]
  for words in slines2: 
    label = int(words[0]) 
    data[label][0] += float(words[-1])     
    data[label][1] += 1
  values = []
  for l in labels:
    d = data[l]
    values.append(d[0] / d[1])
  return FieldOutput(labels = labels, data = values, position = position, dti = dti, dtf = dtf)      


def GetFieldOutput_byRpt(odb, instance, step, frame, original_position, new_position, position, field, sub_field = None, sub_field_prefix = None, sub_set_type = None, sub_set = None, report_name = 'dummy.rpt', dti= 'I', dtf = 'f', delete_report = True):
  '''
  Wraps ``MakeFieldOutputReport`` and ``ReadFieldOutputReport`` in a single function to mimic the behavior ``GetFieldOutput``.
  
  :param odb: output database to be used.
  :type odb: odb instance produced by ``odbAccess.openOdb``
  :param instance: instance to use.
  :type instance: string
  :param step: step to use, this argument can be either the step number or the step label.
  :type step: string or int
  :param frame: frame number, can be negative for reverse counting.
  :type frame: int
  :param original_position: position at which the field is expressed. Can be 'NODAL', 'WHOLE_ELEMENT' or 'INTEGRATION_POINT'.
  :type string:
  :param new_position: position at which you would like the field to be expressed. Can be 'NODAL', 'WHOLE_ELEMENT' or 'INTEGRATION_POINT' or 'ELEMENT_NODAL'. Note that ``ReadFieldOutputReport`` will be capable of averaging values over elements when 'INTEGRATION_POINT' or 'ELEMENT_NODAL' option is sellected.
  :type new_position: string
  :param position: position where the ``FieldOutput`` is to be declared. The function will look at the first and the last column of the report. The first will be considered as the label (*i. e.* element or node) and the last the value. In some case, like reports written using 'ELEMENT_NODAL' or 'INTEGRATION_POINT' locations, each label will appear several times. The present function will collect all the corresponding values and average them. At the end, the only possibilities for this parameter should be 'node' or 'element' as described in the doc of ``FieldOutput``.
  :type position: 'node' or 'element'
  :param field: field to export, example: 'S', 'U', 'EVOL',...
  :type field: string
  :param sub_field: can be a component of an invariant, example: 11, 2, 'Mises', 'Magnitude'. 
  :type sub_field: string or int
  :param sub_set: set to which the report is restricted, must be the label of an existing node or element set.
  :type sub_set: string
  :param sub_set_type: type of the sub_set, can be node or element.
  :type sub_set_type: string
  :param report_name: name or path+name of the report to written.
  :type report_name: string
  :param dti: int data type in array.array
  :type dti: 'I', 'H'
  :param dtf: float data type in array.array
  :type dtf: 'f', 'd'
  :param delete_report: if True, report will be deleted, if false, it will remain.
  :type delete_report: boolean
  :rtype: ``FieldOutput`` instance.
  
  >>> from odbAccess import openOdb
  >>> from abapy.postproc import GetFieldOutput_byRpt
  >>> odb_name = 'indentation.odb'
  >>> odb = openOdb(odb_name)
  >>> S11 = GetFieldOutput_byRpt(
  ...   odb = odb, 
  ...   instance = 'I_SAMPLE', 
  ...   step = 0,
  ...   frame = -1,
  ...   original_position = 'INTEGRATION_POINT', 
  ...   new_position = 'NODAL', 
  ...   position = 'node',
  ...   field = 'S', 
  ...   sub_field = 11, 
  ...   sub_set_type = 'element', 
  ...   sub_set = 'CORE',
  ...   delete_report = False)

  '''
  import os
  MakeFieldOutputReport(
    odb = odb, 
    instance = instance, 
    step = step,
    frame = frame,
    report_name = report_name, 
    original_position = original_position, 
    new_position = new_position, 
    field = field, 
    sub_field = sub_field,
    sub_field_prefix = sub_field_prefix, 
    sub_set_type = sub_set_type, 
    sub_set = sub_set)
  field = ReadFieldOutputReport(
    report_name = report_name, 
    position = position,
    dti = dti,
    dtf = dtf)
  if delete_report:
    try: 
      os.remove(report_name)
    except:
      pass  
  return field

def GetVectorFieldOutput_byRpt(odb, instance, step, frame, original_position, new_position, position, field, sub_field_prefix = None, sub_set_type = None, sub_set = None, report_name = 'dummy.rpt', dti= 'I', dtf = 'f', delete_report = True):
  '''
  Uses ``GetFieldOutput_byRpt`` to produce VectorFieldOutput. 
  
  :param odb: output database to be used.
  :type odb: odb instance produced by ``odbAccess.openOdb``
  :param instance: instance to use.
  :type instance: string
  :param step: step to use, this argument can be either the step number or the step label.
  :type step: string or int
  :param frame: frame number, can be negative for reverse counting.
  :type frame: int
  :param original_position: position at which the field is expressed. Can be 'NODAL', 'WHOLE_ELEMENT' or 'INTEGRATION_POINT'.
  :type string:
  :param new_position: position at which you would like the field to be expressed. Can be 'NODAL', 'WHOLE_ELEMENT' or 'INTEGRATION_POINT' or 'ELEMENT_NODAL'. Note that ``ReadFieldOutputReport`` will be capable of averaging values over elements when 'INTEGRATION_POINT' or 'ELEMENT_NODAL' option is sellected.
  :type new_position: string
  :param position: position where the ``FieldOutput`` is to be declared. The function will look at the first and the last column of the report. The first will be considered as the label (*i. e.* element or node) and the last the value. In some case, like reports written using 'ELEMENT_NODAL' or 'INTEGRATION_POINT' locations, each label will appear several times. The present function will collect all the corresponding values and average them. At the end, the only possibilities for this parameter should be 'node' or 'element' as described in the doc of ``FieldOutput``.
  :type position: 'node' or 'element'
  :param field: field to export, example: 'S', 'U', 'EVOL',...
  :type field: string
  :param sub_set: set to which the report is restricted, must be the label of an existing node or element set.
  :type sub_set: string
  :param sub_set_type: type of the sub_set, can be node or element.
  :type sub_set_type: string
  :param report_name: name or path+name of the report to written.
  :type report_name: string
  :param dti: int data type in array.array
  :type dti: 'I', 'H'
  :param dtf: float data type in array.array
  :type dtf: 'f', 'd'
  :param delete_report: if True, report will be deleted, if false, it will remain.
  :type delete_report: boolean
  :rtype: ``VectorFieldOutput`` instance.
  
  >>> from odbAccess import openOdb
  >>> from abapy.postproc import GetVectorFieldOutput_byRpt
  >>> odb_name = 'indentation.odb'
  >>> odb = openOdb(odb_name)
  >>> U = GetVectorFieldOutput_byRpt(
  ...   odb = odb, 
  ...   instance = 'I_SAMPLE', 
  ...   step = 0,
  ...   frame = -1,
  ...   original_position = 'NODAL', 
  ...   new_position = 'NODAL', 
  ...   position = 'node',
  ...   field = 'U', 
  ...   sub_set_type = 'element', 
  ...   sub_set = 'CORE',
  ...   delete_report = True)

  '''
  
  field1 =GetFieldOutput_byRpt(
  odb = odb, 
  instance = instance, 
  step = step,
  frame = frame,
  original_position = original_position, 
  new_position = new_position, 
  position = position,
  field = field,
  sub_field_prefix = sub_field_prefix, 
  sub_field = 1, 
  sub_set_type = sub_set_type, 
  sub_set = sub_set,
  delete_report = delete_report,
  dti = dti,
  dtf = dtf)
  
  field2 =GetFieldOutput_byRpt(
  odb = odb, 
  instance = instance, 
  step = step,
  frame = frame,
  original_position = original_position, 
  new_position = new_position, 
  position = position,
  field = field,
  sub_field_prefix = sub_field_prefix,  
  sub_field = 2, 
  sub_set_type = sub_set_type, 
  sub_set = sub_set,
  delete_report = delete_report,
  dti = dti,
  dtf = dtf)
  
  if type(step) == int:
    stepLabel = odb.steps.keys()[step]
  else:
    stepLabel = step # Merci Jonathan !
  componentLabels = odb.steps[stepLabel].frames[frame].fieldOutputs[field].componentLabels
  if len(componentLabels) == 3:
    field3 =GetFieldOutput_byRpt(
    odb = odb, 
    instance = instance, 
    step = step,
    frame = frame,
    original_position = original_position, 
    new_position = new_position, 
    position = position,
    field = field,
    sub_field_prefix = sub_field_prefix,  
    sub_field = 3, 
    sub_set_type = sub_set_type, 
    sub_set = sub_set,
    delete_report = delete_report,
    dti = dti,
    dtf = dtf)
  
    vector_field = VectorFieldOutput(
      data1 = field1,
      data2 = field2,
      data3 = field3,
      position = position,
      dti = dti,
      dtf = dtf)
  else:
    vector_field = VectorFieldOutput(
      data1 = field1,
      data2 = field2,
      position = position,
      dti = dti,
      dtf = dtf)
  return vector_field
  

def GetTensorFieldOutput_byRpt(odb, instance, step, frame, original_position, new_position, position, field, sub_field_prefix = None, sub_set_type = None, sub_set = None, report_name = 'dummy.rpt', dti= 'I', dtf = 'f', delete_report = True):
  '''
  Uses ``GetFieldOutput_byRpt`` to produce TensorFieldOutput. 
  
  :param odb: output database to be used.
  :type odb: odb instance produced by ``odbAccess.openOdb``
  :param instance: instance to use.
  :type instance: string
  :param step: step to use, this argument can be either the step number or the step label.
  :type step: string or int
  :param frame: frame number, can be negative for reverse counting.
  :type frame: int
  :param original_position: position at which the field is expressed. Can be 'NODAL', 'WHOLE_ELEMENT' or 'INTEGRATION_POINT'.
  :type string:
  :param new_position: position at which you would like the field to be expressed. Can be 'NODAL', 'WHOLE_ELEMENT' or 'INTEGRATION_POINT' or 'ELEMENT_NODAL'. Note that ``ReadFieldOutputReport`` will be capable of averaging values over elements when 'INTEGRATION_POINT' or 'ELEMENT_NODAL' option is sellected.
  :type new_position: string
  :param position: position where the ``FieldOutput`` is to be declared. The function will look at the first and the last column of the report. The first will be considered as the label (*i. e.* element or node) and the last the value. In some case, like reports written using 'ELEMENT_NODAL' or 'INTEGRATION_POINT' locations, each label will appear several times. The present function will collect all the corresponding values and average them. At the end, the only possibilities for this parameter should be 'node' or 'element' as described in the doc of ``FieldOutput``.
  :type position: 'node' or 'element'
  :param field: field to export, example: 'S', 'U', 'EVOL',...
  :type field: string
  :param sub_set: set to which the report is restricted, must be the label of an existing node or element set.
  :type sub_set: string
  :param sub_set_type: type of the sub_set, can be node or element.
  :type sub_set_type: string
  :param report_name: name or path+name of the report to written.
  :type report_name: string
  :param dti: int data type in array.array
  :type dti: 'I', 'H'
  :param dtf: float data type in array.array
  :type dtf: 'f', 'd'
  :param delete_report: if True, report will be deleted, if false, it will remain.
  :type delete_report: boolean
  :rtype: ``TensorFieldOutput`` instance.
  
  >>> from odbAccess import openOdb
  >>> from abapy.postproc import GetTensorFieldOutput_byRpt
  >>> odb_name = 'indentation.odb'
  >>> odb = openOdb(odb_name)
  >>> S = GetTensorFieldOutput_byRpt(
  ...   odb = odb, 
  ...   instance = 'I_SAMPLE', 
  ...   step = 0,
  ...   frame = -1,
  ...   original_position = 'INTEGRATION_POINT', 
  ...   new_position = 'NODAL', 
  ...   position = 'node',
  ...   field = 'S', 
  ...   sub_set_type = 'element', 
  ...   sub_set = 'CORE',
  ...   delete_report = True)


  '''
  
  field11 =GetFieldOutput_byRpt(
  odb = odb, 
  instance = instance, 
  step = step,
  frame = frame,
  original_position = original_position, 
  new_position = new_position, 
  position = position,
  field = field,
  sub_field_prefix = sub_field_prefix,  
  sub_field = 11, 
  sub_set_type = sub_set_type, 
  sub_set = sub_set,
  delete_report = delete_report,
  dti = dti,
  dtf = dtf)
  
  field22 =GetFieldOutput_byRpt(
  odb = odb, 
  instance = instance, 
  step = step,
  frame = frame,
  original_position = original_position, 
  new_position = new_position, 
  position = position,
  field = field, 
  sub_field_prefix = sub_field_prefix, 
  sub_field = 22, 
  sub_set_type = sub_set_type, 
  sub_set = sub_set,
  delete_report = delete_report,
  dti = dti,
  dtf = dtf)
  
  field33 =GetFieldOutput_byRpt(
  odb = odb, 
  instance = instance, 
  step = step,
  frame = frame,
  original_position = original_position, 
  new_position = new_position, 
  position = position,
  field = field, 
  sub_field_prefix = sub_field_prefix, 
  sub_field = 33, 
  sub_set_type = sub_set_type, 
  sub_set = sub_set,
  delete_report = delete_report,
  dti = dti,
  dtf = dtf)
  
  field12 =GetFieldOutput_byRpt(
  odb = odb, 
  instance = instance, 
  step = step,
  frame = frame,
  original_position = original_position, 
  new_position = new_position, 
  position = position,
  field = field, 
  sub_field_prefix = sub_field_prefix, 
  sub_field = 12, 
  sub_set_type = sub_set_type, 
  sub_set = sub_set,
  delete_report = delete_report,
  dti = dti,
  dtf = dtf)
  
  if type(step) == int:
    stepLabel = odb.steps.keys()[step]
  else:
    stepLabel = step # Merci Jonathan !
  componentLabels = odb.steps[stepLabel].frames[frame].fieldOutputs[field].componentLabels
  if len(componentLabels) == 6:
     
    field13 =GetFieldOutput_byRpt(
    odb = odb, 
    instance = instance, 
    step = step,
    frame = frame,
    original_position = original_position, 
    new_position = new_position, 
    position = position,
    field = field, 
    sub_field_prefix = sub_field_prefix, 
    sub_field = 13, 
    sub_set_type = sub_set_type, 
    sub_set = sub_set,
    delete_report = delete_report,
    dti = dti,
    dtf = dtf)
    
    field23 =GetFieldOutput_byRpt(
    odb = odb, 
    instance = instance, 
    step = step,
    frame = frame,
    original_position = original_position, 
    new_position = new_position, 
    position = position,
    field = field, 
    sub_field_prefix = sub_field_prefix, 
    sub_field = 23, 
    sub_set_type = sub_set_type, 
    sub_set = sub_set,
    delete_report = delete_report,
    dti = dti,
    dtf = dtf)
  
    tensor_field = TensorFieldOutput(
      data11 = field11,
      data22 = field22,
      data33 = field33,
      data12 = field12,
      data13 = field13,
      data23 = field23,
      position = position,
      dti = dti,
      dtf = dtf)
  else:
    tensor_field = TensorFieldOutput(
      data11 = field11,
      data22 = field22,
      data33 = field33,
      data12 = field12,
      position = position,
      dti = dti,
      dtf = dtf)
  return tensor_field
  
class FieldOutput(object):
  '''
  Scalar output representing a field evaluated on nodes or elements referenced by their labels. A FieldOutput instance cannot be interpreted with its mesh. On initiation, labels and data will be reordered to have labels sorted. 
 
  :param position: location of the field evaluation
  :type position: 'node' or 'element'
  :param data: value of the field where it is evaluated
  :type data: list, array.array, numpy.array containing floats
  :param labels: labels of the nodes/elements where the field is evaluated. If None, labels will be [1,2,...,len(data)+1] 
  :type labels: list, array.array, numpy.array containt ints or None. 
  :param dti: int data type in array.array
  :type dti: 'I', 'H'
  :param dtf: float data type in array.array
  :type dtf: 'f', 'd'
  
  >>> from abapy.postproc import FieldOutput
  >>> data = [-1.,5.,3.]
  >>> labels = [1,3,2]
  >>> fo = FieldOutput(data=data, labels = labels, position = 'node')
  >>> print fo # data is sorted by labels
  FieldOutput instance
  Position = node
  Label	Data
  1	-1.0
  2	3.0
  3	5.0
  >>> print fo[1:2] # slicing
  FieldOutput instance
  Position = node
  Label	Data
  1	-1.0
  >>> print fo[2] # indexing
  FieldOutput instance
  Position = node
  Label	Data
  2	3.0
  >>> print fo[1,3] # multiple indexing
  FieldOutput instance
  Position = node
  Label	Data
  1	-1.0
  3	5.0
  >>> print fo*2 # multiplication
  FieldOutput instance
  Position = node
  Label	Data
  1	-2.0
  2	6.0
  3	10.0
  >>> fo2 = fo**2  #power
  >>> print fo2
  FieldOutput instance
  Position = node
  Label	Data
  1	1.0
  2	9.0
  3	25.0
  >>> print fo * fo2
  FieldOutput instance
  Position = node
  Label	Data
  1	-1.0
  2	27.0
  3	125.0
  >>> print fo + fo2
  FieldOutput instance
  Position = node
  Label	Data
  1	0.0
  2	12.0
  3	30.0
  >>> print abs(fo) 
  FieldOutput instance
  Position = node
  Label	Data
  1	1.0
  2	3.0
  3	5.0

  .. note::
       If dti='H' is chosen, labels are stored as unsigned 16 bits ints. If more than 65k labels are stored, an OverFlow error will be raised.
       
  '''
  def __init__(self,position='node',data=None,labels=None,dti='I',dtf='f'):
    from array import array
    self.dti, self.dtf = dti, dtf
    self.position = position
    self.data = array(dtf,[])
    self.labels = array(dti,[])
    if data != None:
      if labels == None:
        labels = range(1,len(data)+1)
      else:
        if len(labels) != len(data) : raise Exception, 'labels and data must have the same length'
        if min(labels) < 1: raise Exception, 'labels must be int > 0'
        zipped = zip(labels,data)
        labels, data = zip(*sorted(zipped))
      self.data = array(dtf,data)
      self.labels = array(dti,labels) 
      '''
      if len(data) == len(labels):
        for i in xrange(len(labels)):
          self.add_data(labels[i],data[i])
      '''
      
  def add_data(self,label, data):
    '''
    Adds one point to a FieldOutput instance. Label must not already exist in the current FieldOutput, if not so, nothing will be changed. Data and label will be inserted in self.data, self.labels in order to keep self.labels sorted.
   
   >>> from abapy.postproc import FieldOutput
   >>> data = [5.5, 2.2]
   >>> labels = [1,4]
   >>> temperature = FieldOutput(labels = labels, data = data, position = 'node')
   >>> temperature.add_data(2, 5.)
   >>> temperature.data # labels are sorted
   array('f', [5.5, 5.0, 2.200000047683716])
   >>> temperature.labels # data was sorted like labels
   array('I', [1L, 2L, 4L]) 

   
  
 
  :param label: labels of the nodes/elements where the field is evaluated.
  :type labels: int > 0
  :param data: value of the field where it is evaluated
  :type data: float
  
  '''
    from copy import deepcopy
    from array import array
    if label in self.labels:
      print 'data already exists at this node'
      return 
    else:
      self_data, self_labels = self.data, self.labels
      self_data.append(data)
      self_labels.append(label)
      zipped = zip(self_labels,self_data)
      labels, data = zip(*sorted(zipped))
      self.data = array(self.dtf,data)
      self.labels = array(self.dti,labels)  
      
  def dump2vtk(self,name='fieldOutput', header = True):
    '''
    Converts the FieldOutput instance to VTK format which can be directly read by Mayavi2 or Paraview. This method is very useful to quickly and efficiently plot 3D mesh and fields.
    
    :param name: name used for the field in the output.
    :type name: string
    :param header: if True, adds the location header (eg. CELL_DATA / POINT_DATA)
    :type header: boolean
    :rtype: string
    
    .. plot:: example_code/postproc/FieldOutput-dump2vtk.py
     :include-source:
    
    Result in Paraview:
    
    .. image:: example_code/postproc/FieldOutput-dump2vtk.svg
       :width: 750
    '''
    out = ""
    ld = len(self.data)
    if header:
      if self.position == 'node': dataType = 'POINT_DATA'
      if self.position == 'element': dataType = 'CELL_DATA' 
      out += "{0} {1}\n".format(dataType, ld)
    out += 'SCALARS {0} float 1\nLOOKUP_TABLE default\n'.format(name)
    pattern = '{0}\n'
    for d in self.data:
      out += pattern.format(d)
    return out
  
 
  def __repr__(self):
    l = len(self.labels)
    return '<FieldOutput instance: {0} locations>'.format(l)
  
  def __getitem__(self,s):
    from array import array
    from copy import deepcopy
    from numpy import nan
    labs = []
    if type(s) in [int, long]:
      labs = [s]
    if type(s) is slice:
      start = s.start
      stop  = s.stop
      step  = s.step
      if step == None: step = 1
      labs = range(start,stop,step)
    if type(s) in [tuple,list, array]:  
      for a in s:
        if type(a) in [int, long]:labs.append(a)
       
    labels = self.labels
    dtf = self.dtf
    dti = self.dti
    data = self.data
    position = self.position
    fo = FieldOutput(position = position, dti = dti, dtf = dtf)
    for l in labs:    
      if l in labels:
        i = labels.index(l)
        fo.add_data(label = l, data = data[i])
      else:
        fo.add_data(label = l, data = nan)  
    return fo     
  
  def __str__(self):
    labels, data, position = self.labels, self.data, self.position
    out = 'FieldOutput instance\nPosition = {0}\nLabel\tData\n'.format(position)
    pattern = '{0}\t{1}\n'
    for i in xrange(len(labels)): out += pattern.format(labels[i], data[i])
    return out
  
  def get_data(self,label):
    '''
    Returns data at a location with given label. 
    
    :param label: location's label.
    :type label: int > 0
    :rtype: float
        
    .. note:: Requesting data at a label that does not exist in the instance will just lead in a warning but if label is negative or is not int, then an Exception will be raised.
    '''
     
    if type(label) not in [int, long] or label <= 0:
      raise Exception, 'label must be int > 0, got {0}'.format(label)  
    if label in self.labels:
      i = self.labels.index(label)
      return self.data[i] 
    else:
      print 'Info: requesting data at non existant location, returning None'    
  
  def _PreProcess(self,other=None):
    '''
    Preprocesses internal operations such as ``__add__``, ``__mul__`` by checking position, labels dtype compatibility and preparing operands to fasten computations. Other can be None, in this case no checking are performed and self.data preprocessing is still perfomed.
    
    .. note:: uses Numpy.
    '''
    from copy import copy
    from array import array as a_array
    from numpy import array as n_array
    from numpy import float32, float64, uint16, uint32
    if other == None : other = 1. 
    if isinstance(other,FieldOutput):
      if self.labels != other.labels: raise Exception, 'operands labels must be identicals.'
      if self.position != other.position: raise Exception, 'operands position must be identicals.'
      if self.dti == 'H' and other.dti == 'H':
        dti = 'H'
      else:
        dti = 'I'
      if self.dtf == 'd' or other.dtf == 'd':
        dtf = 'd'
      else: 
        dtf = 'f'
      out = FieldOutput(dti=dti, dtf=dtf, position=self.position)
      other_data = other.data
    if type(other) in [float, int, long]:
      dti, dtf = self.dti, self.dtf
      out = FieldOutput(position = self.position, dti = dti, dtf = dtf)
      other_data = float(other)
    if dtf == 'f': ndtf = float32  
    if dtf == 'd': ndtf = float64    
    other_data = n_array(other_data,ndtf)
    self_data = n_array(self.data,ndtf)
    return out, self_data, other_data
  
  def _PostProcess(self, out, new_data):
    '''
    Postprocesses internal operations such as ``__add__``, ``__mul__``.
    
    .. note:: uses Numpy.
    '''
    
    from array import array
    out.labels = self.labels
    out.data = array( out.dtf, new_data.tolist() )
    return out
  
  def __add__(self,other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput)) == False:
      return NotImplemented
    out, self_data, other_data = self._PreProcess(other)
    new_data = self_data + other_data 
    out = self._PostProcess(out,new_data)
    return out 
    
  __radd__ = __add__ 
  
  def __sub__(self,other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput)) == False:
      return NotImplemented
    out, self_data, other_data = self._PreProcess(other)
    new_data = self_data - other_data 
    out = self._PostProcess(out,new_data)
    return out 
  
  def __rsub__(self,other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput)) == False:
      return NotImplemented
    return self * -1 + other 
  
  def __rsub__(self, other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput)) == False:
      return NotImplemented
    return -self + other 
    
  def __mul__(self,other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput)) == False:
      return NotImplemented
    out, self_data, other_data = self._PreProcess(other)
    new_data = self_data * other_data 
    out = self._PostProcess(out,new_data)
    return out  
  
  __rmul__ = __mul__  
    
  def __div__(self,other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput)) == False:
      return NotImplemented
    out, self_data, other_data = self._PreProcess(other)
    new_data = self_data / other_data 
    out = self._PostProcess(out,new_data)
    return out 
  
  def __rdiv__(self, other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput)) == False:
      return NotImplemented
    return other *  self**-1  
    
  def __pow__(self,other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput)) == False:
      return NotImplemented
    out, self_data, other_data = self._PreProcess(other)
    new_data = self_data ** other_data 
    out = self._PostProcess(out,new_data)
    return out 
      
  def __neg__(self):
    out, self_data, other_data = self._PreProcess(-1.)
    new_data = self_data * other_data 
    out = self._PostProcess(out,new_data)
    return out 
  
  def __abs__(self):
    out, self_data, other_data = self._PreProcess()
    new_data = abs(self_data)
    out = self._PostProcess(out,new_data)
    return out 
  
     
def ZeroFieldOutput_like(fo):
  '''
  A FieldOutput containing only zeros but with the same position, labels and dtypes as the input. 
 
  :param fo: field output to be used.
  :type fo: FieldOutput instance
  :rtype: FieldOutput instance
  
  .. note:: uses Numpy.
  '''
  from copy import copy
  from array import array
  #from numpy import zeros_like # LC 12/04/2012: commented to use a non numpy method instead.
  if isinstance(fo,FieldOutput) == False:
    raise Exception, 'input must be FieldOutput instance.'
  dti, dtf, position = fo.dti, fo.dtf, fo.position
  #data = array(dtf, zeros_like(fo.data).tolist()) # LC 12/04/2012: commented to use a non numpy method instead.
  data = array(dtf, [ 0. for i in fo.data])
  labels = copy(fo.labels)
  return FieldOutput(position=position, data=data, labels=labels, dti=dti, dtf=dtf)
  
def OneFieldOutput_like(fo):
  '''
  A FieldOutput containing only ones but with the same position, labels and dtypes as the input. 
 
  :param fo: field output to be used.
  :type fo: FieldOutput instance
  :rtype: FieldOutput instance
  
  .. note:: uses Numpy.
  '''
  from copy import copy
  from array import array
  from numpy import ones_like
  if isinstance(fo,FieldOutput) == False:
    raise Exception, 'input must be FieldOutput instance.'
  dti, dtf, position = fo.dti, fo.dtf, fo.position
  data = array(dtf, ones_like(fo.data).tolist())
  labels = copy(fo.labels)
  return FieldOutput(position=position, data=data, labels=labels, dti=dti, dtf=dtf)
  
  

class VectorFieldOutput:
  '''
  3D vector field output. Using this class instead of 3 scalar FieldOutput instances is efficient because labels are stored only since and allows all vector operations like dot, cross, norm.
 
  :param data1: x coordinate
  :type data1: FieldOutput instance or None
  :param data2: y coordinate
  :type data2: FieldOutput instance or None
  :param data3: z coordinate
  :type data3: FieldOutput instance or None
  :param position: position at which data is computed
  :type position: 'node' or 'element'
  :param dti: array.array int data type
  :type dti: 'I' for uint32 or 'H' for uint16
  :param dtf: array.array int data type
  :type dtf: 'f' float32 or 'd' for float64
  
  >>> from abapy.postproc import FieldOutput, VectorFieldOutput
  >>> data1 = [1,2,3,5,6,0]
  >>> data2 = [1. for i in data1]
  >>> labels = range(1,len(data1)+1)
  >>> fo1, fo2 = FieldOutput(labels = labels, data=data1, position='node' ), FieldOutput(labels = labels, data=data2,position='node')
  >>> vector = VectorFieldOutput(data1 = fo1, data2 = fo2 )
  >>> vector2 = VectorFieldOutput(data2 = fo2 )
  >>> vector # short description
  <VectorFieldOutput instance: 6 locations>
  >>> print vector # long description
  VectorFieldOutput instance
  Position = node
  Label	Data1	Data2	Data3
  1	1.0	1.0	0.0
  2	2.0	1.0	0.0
  3	3.0	1.0	0.0
  4	5.0	1.0	0.0
  5	6.0	1.0	0.0
  6	0.0	1.0	0.0
  >>> print vector[6] # Returns a VectorFieldOutput instance
  VectorFieldOutput instance
  Position = node
  Label	Data1	Data2	Data3
  6	0.0	1.0	1.0
  >>> print vector[1,4,6] # Picking label by label
  VectorFieldOutput instance
  Position = node
  Label	Data1	Data2	Data3
  1	1.0	1.0	1.0
  4	5.0	1.0	1.0
  6	0.0	1.0	1.0
  >>> print vector[1:6:2] # Slicing
  VectorFieldOutput instance
  Position = node
  Label	Data1	Data2	Data3
  1	1.0	1.0	1.0
  3	3.0	1.0	1.0
  5	6.0	1.0	1.0
  >>> vector.get_data(6) # Returns 3 floats
  (0.0, 1.0, 0.0)
  >>> vector.norm() # Returns norm
  <FieldOutput instance: 6 locations>
  >>> vector.sum() # returns the sum of coords
  <FieldOutput instance: 6 locations>
  >>> vector * vector2 # Itemwise product (like numpy, unlike matlab)
  <VectorFieldOutput instance: 6 locations>
  >>> vector.dot(vector2) # Dot/Scalar product
  <FieldOutput instance: 6 locations>
  >>> vector.cross(vector2) # Cross/Vector product
  <VectorFieldOutput instance: 6 locations>
  >>> vector + 2 # Itemwise addition
  <VectorFieldOutput instance: 6 locations>
  >>> vector * 2 # Itemwise multiplication
  <VectorFieldOutput instance: 6 locations>
  >>> vector / 2 # Itemwise division
  <VectorFieldOutput instance: 6 locations>
  >>> vector / vector2 # Itemwise division between vectors (numpy way)
  Warning: divide by zero encountered in divide
  Warning: invalid value encountered in divide
  <VectorFieldOutput instance: 6 locations>
  >>> abs(vector) # Absolute value
  <VectorFieldOutput instance: 6 locations>
  >>> vector ** 2 # Power
  <VectorFieldOutput instance: 6 locations>
  >>> vector ** vector # Itemwize power
  <VectorFieldOutput instance: 6 locations>
 
  .. note::
       * data1, data2 and data3 must have same position and label or be None. If one data is None, it is supposed to be zero. 
       * Storage data dype is the highest standard of all 3 data.
       * Numpy is not used in the constructor to allow the creation of instances in Abaqus Python but most other operation require numpy for speed reasons.'''
       
  def __init__(self,data1=None,data2=None,data3=None, position = 'node', dti='I', dtf = 'f'):
    from array import array
    self.position = position
    self.dti = dti
    self.dtf = dtf
    self.labels = array(dti,[])
    self.data1, self.data2, self.data3 = array(dtf,[]), array(dtf,[]), array(dtf,[])
    data = [data1, data2, data3]
    isNone = [data1 == None, data2 == None, data3 == None]
    for d in data:
      if (isinstance(d,FieldOutput) == False) and (d != None):
        raise Exception, 'data1, data2 and data3 must be FieldOutput instances.'
    
    # Remove comments when python 2.4 is not used anymore
    if isNone != [True, True, True]:
      for i in [2,1,0]:
        d = data[i]
        if d != None: refData = d
      labels = refData.labels
      useShortInts = True
      positions = []
      for i in xrange(3):
        if data[i] == None: data[i] = ZeroFieldOutput_like(refData)
        if data[i].labels != labels:
          raise Exception, 'data1, data2 and data3 must be fieldOutputs sharing the same labels.'    
        if data[i].dtf == 'd': self.dtf == 'd'
        if data[i].dti == 'I': useShortInts = False
        positions.append(data[i].position)
      if useShortInts: self.dti = 'H'            
      if len(set(positions)) > 1:
        raise Exception, 'inputs must have the same position or be None.'
      '''
      for i in xrange(len(labels)):
        l = labels[i]
        d1, d2, d3 = data[0].data[i] , data[1].data[i] , data[2].data[i]
        self.add_data(label = l, data1 = d1, data2 = d2, data3 = d3)
      '''
      self.labels = labels
      self.data1, self.data2, self.data3 = data[0].data, data[1].data, data[2].data
    
  def norm(self):
    '''
    Computes norm of the vector at each location and returns it as a scalar FieldOutput.
    
    >>> norm = Vec.norm()
    
    :rtype: FieldOutput instance
    '''
    return (( self * self )**.5).sum()
     
  def get_coord(self,number):
    '''
    Returns a coordinate of the vector as a FieldOutput.
       
    :param number: requested coordinate number, 1 is x and so on.
    :type number: 1,2 or 3  
    :rtype: FieldOutput instance
    
    >>> v1 = Vec.get_coord(1)
    '''
    import numpy as np
    #from copy import copy
    if number not in [1,2,3]:
      raise Exception, 'number must be 1,2 or 3'  
    dti, dtf = self.dti, self.dtf
    position = self.position
    if number == 1: data = self.data1
    if number == 2: data = self.data2
    if number == 3: data = self.data3
    return FieldOutput(position = position, data = data, labels=self.labels, dti=dti, dtf=dtf )
    
  def get_data(self,label):
    '''
    Returns coordinates at a location with given label. 
    
    :param label: location's label.
    :type label: int > 0
    :rtype: float, float, float
        
    .. note:: Requesting data at a label that does not exist in the instance will just lead in a warning but if label is negative or is not int, then an Exception will be raised.
    '''
     
    if type(label) not in [int, long] or label <= 0:
      raise Exception, 'label must be int > 0, got {0}'.format(label)  
    if label in self.labels:
      i = self.labels.index(label)
      return self.data1[i], self.data2[i], self.data3[i] 
    else:
      print 'Info: requesting data at non existant location, returning None'
 
  def __repr__(self):
    l = len(self.labels)
    return '<VectorFieldOutput instance: {0} locations>'.format(l)
  def __getitem__(self,s):
    from array import array
    from copy import deepcopy
    from numpy import nan
    labs = []
    if type(s) in [int, long]:
      labs = [s]
    if type(s) is slice:
      start = s.start
      stop  = s.stop
      step  = s.step
      labs = range(start,stop,step)
    if type(s) in [tuple,list, array]:  
      for a in s:
        if type(a) in [int, long]:labs.append(a)
       
    labels = self.labels
    dtf = self.dtf
    dti = self.dti
    data1, data2, data3 = self.data1, self.data2, self.data3
    position = self.position
    fo = VectorFieldOutput(position = position, dti = dti, dtf = dtf)
    for l in labs:    
      if l in labels:
        i = labels.index(l)
        fo.add_data(label = l, data1 = data1[i], data2 = data2[i], data3 = data3[i] )
      else:
        fo.add_data(label = l, data1 = nan, data2 = nan, data3 = nan )
    return fo     
  def __str__(self):
    labels, position = self.labels, self.position
    data1, data2, data3 =  self.data1, self.data2, self.data3
    out = 'VectorFieldOutput instance\nPosition = {0}\nLabel\tData1\tData2\tData3\n'.format(position)
    pattern = '{0}\t{1}\t{2}\t{3}\n'
    for i in xrange(len(labels)): out += pattern.format(labels[i], data1[i], data2[i], data3[i])
    return out
  def add_data(self,label, data1=0., data2=0., data3=0.):
    '''
    Adds one point to a VectorFieldOutput instance. Label must not already exist in the current FieldOutput, if not so, nothing will be changed. Data and label will be inserted in self.data, self.labels in order to keep self.labels sorted.
   
    :param label: labels of the nodes/elements where the field is evaluated.
    :type labels: int > 0
    :param data1: value of the coordinate 1 of the field where it is evaluated.
    :type data1: float
    :param data2: value of the coordinate 2 of the field where it is evaluated.
    :type data2: float
    :param data3: value of the coordinate 3 of the field where it is evaluated.
    :type data3: float
    '''
    '''
    from copy import deepcopy
    if label in self.labels:
      print 'data already exists at this node'
      return 
    if len(self.labels) != 0:  
      if label > max(self.labels):
        self.data1.append(data1)
        self.data2.append(data2)
        self.data3.append(data3)
        self.labels.append(label)
      else:
        lab = deepcopy(self.labels)
        lab.append(label)
        lab = sorted(lab)
        indice = lab.index(label)
        self.labels.insert(indice,label)
        self.data1.insert(indice,data1)
        self.data2.insert(indice,data2)
        self.data3.insert(indice,data3)
    else:
      self.data1.append(data1)
      self.data2.append(data2)
      self.data3.append(data3)
      self.labels.append(label)
    '''
    from array import array
    dti,dtf = self.dti, self.dtf
    if label in self.labels:
      print 'data already exists at this node'
      return 
    else:
      self_data1, self_data2, self_data3, self_labels = self.data1, self.data2, self.data3, self.labels
      self_data1.append(data1)
      self_data2.append(data2)
      self_data3.append(data3)
      self_labels.append(label)
      zipped = zip(self_labels,self_data1, self_data2, self_data3)
      labels, data1, data2, data3 = zip(*sorted(zipped))
      self.data1 = array(self.dtf,data1)
      self.data2 = array(self.dtf,data2)
      self.data3 = array(self.dtf,data3)
      self.labels = array(self.dti,labels)   
  
  def dump2vtk(self,name='vectorFieldOutput', header = True):
    '''
    Converts the VectorFieldOutput instance to VTK format which can be directly read by Mayavi2 or Paraview. This method is very useful to quickly and efficiently plot 3D mesh and fields.
    
    :param name: name used for the field in the output.
    :type name: string
    :rtype: string
    
    >>> from abapy.postproc import FieldOutput, VectorFieldOutput
    >>> from abapy.mesh import RegularQuadMesh
    >>> mesh = RegularQuadMesh()
    >>> data1 = [2,2,5,10]
    >>> data2 = [1. for i in data1]
    >>> labels = range(1,len(data1)+1)
    >>> fo1, fo2 = FieldOutput(labels = labels, data=data1, position='node' ), FieldOutput(labels = labels, data=data2,position='node')
    >>> vector = VectorFieldOutput(data1 = fo1, data2 = fo2 )
    >>> out = mesh.dump2vtk() + vector.dump2vtk()
    >>> f = open('vector.vtk','w')
    >>> f.write(out)
    >>> f.close()
    '''
    d1, d2, d3 = self.data1, self.data2, self.data3
    ld = len(d1)
    out = ""
    if header:
      if self.position == 'node': dataType = 'POINT_DATA'
      if self.position == 'element': dataType = 'CELL_DATA' 
      out += "{0} {1}\n".format(dataType, ld)
    out += 'VECTORS {0} float\n'.format(name)
    pattern = '{0} {1} {2}\n'
    for i in xrange(ld):
      out += pattern.format(d1[i], d2[i], d3[i])
    return out
   
  def _PreProcess(self, other):
    s1, s2, s3 = self.get_coord(1), self.get_coord(2), self.get_coord(3)
    if isinstance(other, VectorFieldOutput):
      o1, o2, o3 = other.get_coord(1), other.get_coord(2), other.get_coord(3)
    if isinstance(other, FieldOutput):
      o1, o2, o3 = other, other, other
    if type(other) in [float, int, long]:
      o = other * OneFieldOutput_like(self.get_coord(1))
      o1, o2, o3  = o, o, o
    return s1, s2, s3, o1, o2, o3
     
  def _PostProcess(self, out1, out2, out3):
    out = VectorFieldOutput(data1 = out1, data2 = out2, data3 = out3)
    return out
     
  def __add__(self, other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput) or isinstance(other, VectorFieldOutput)) == False:
      return NotImplemented
    s1, s2, s3, o1, o2, o3 = self._PreProcess(other)
    out1, out2, out3 = s1 + o1, s2 + o2, s3 + o3
    return self._PostProcess(out1, out2, out3)
  
  __radd__ = __add__
      
  def __mul__(self, other): # term wise product
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput) or isinstance(other, VectorFieldOutput)) == False:
      return NotImplemented
    s1, s2, s3, o1, o2, o3 = self._PreProcess(other)
    out1, out2, out3 = s1 * o1, s2 * o2, s3 * o3
    return self._PostProcess(out1, out2, out3)
    
  __rmul__ = __mul__
  
  def dot(self, other): # dot (scalar) product
    '''
    Returns the dot (*i. e.* scalar) product of two vector field outputs.
    
    :param other: Another vector field
    :type other: ``VectorFieldOutput``
    :rtype: ``FieldOutput``
    '''
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput) or isinstance(other, VectorFieldOutput)) == False:
      return NotImplemented
    return (self * other).sum()
    
  def __neg__(self):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput) or isinstance(other, VectorFieldOutput)) == False:
      return NotImplemented
    other = 1.
    s1, s2, s3 = self.get_coord(1), self.get_coord(2), self.get_coord(3)
    return VectorFieldOutput(data1 = -s1, data2 = -s2, data3 = -s3)
    
  def __sub__(self, other):
    if type(other) in [int, float, long] or isinstance(other, FieldOutput) or isinstance(other, VectorFieldOutput) == False:
      return NotImplemented
    return self + (-other)   
  
  def __rsub__(self, other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput) or isinstance(other, VectorFieldOutput)) == False:
      return NotImplemented
    return -self + other
    
    
  def __div__(self, other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput) or isinstance(other, VectorFieldOutput)) == False:
      return NotImplemented
    s1, s2, s3, o1, o2, o3 = self._PreProcess(other)
    out1, out2, out3 = s1 / o1, s2 / o2, s3 / o3
    return self._PostProcess(out1, out2, out3)
    
  def __rdiv__(self, other):
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput) or isinstance(other, VectorFieldOutput)) == False:
      return NotImplemented
    return other * self**-1
    
  def cross(self, other): # cross (vector) product
    '''
    Returns the cross product of two vector field outputs.
    
    :param other: Another vector field
    :type other: ``VectorFieldOutput``
    :rtype: ``VectorFieldOutput``
    '''
    if isinstance(other, VectorFieldOutput) == False:
      return NotImplemented
    s1, s2, s3, o1, o2, o3 = self._PreProcess(other)
    out1 = s2 * o3 - s3 * o2
    out2 = s3 * o1 - s1 * o3
    out3 = s1 * o2 - s2 * o1
    return self._PostProcess(out1, out2, out3)
  
  def __pow__(self, other): 
    if (type(other) in [int, float, long] or isinstance(other, FieldOutput) or isinstance(other, VectorFieldOutput)) == False:
      return NotImplemented
    s1, s2, s3, o1, o2, o3 = self._PreProcess(other)
    out1, out2, out3 = s1 ** o1, s2 ** o2, s3 ** o3
    return self._PostProcess(out1, out2, out3)
    
  def __abs__(self): 
    s1, s2, s3 = self.get_coord(1), self.get_coord(2), self.get_coord(3)
    return VectorFieldOutput(data1 = abs(s1), data2 = abs(s2), data3 = abs(s3))
     
  def sum(self):
    '''
    Returns the sum of all coordinates.
    
    :rtype: FieldOutput instance
    '''
    return self.get_coord(1) + self.get_coord(2) + self.get_coord(3)
    
    
class TensorFieldOutput:
  '''
  Symmetric tensor field output. Using this class instead of 6 scalar FieldOutput instances is efficient because labels are stored only since and allows all operations like invariants, product, sum...
 
  :param data11: 11 component
  :type data11: FieldOutput instance or None
  :param data22: 22 component
  :type data22: FieldOutput instance or None
  :param data33: 33 component
  :type data33: FieldOutput instance or None
  :param data12: 12 component
  :type data12: FieldOutput instance or None
  :param data13: 13 component
  :type data13: FieldOutput instance or None
  :param data23: 23 component
  :type data23: FieldOutput instance or None
  :param position: position at which data is computed
  :type position: 'node' or 'element'
  :param dti: array.array int data type
  :type dti: 'I' for uint32 or 'H' for uint16
  :param dtf: array.array int data type
  :type dtf: 'f' float32 or 'd' for float64
  
  >>> from abapy.postproc import FieldOutput, TensorFieldOutput, VectorFieldOutput
  >>> data11 = [1., 1., 1.]
  >>> data22 = [2., 4., -1]
  >>> data12 = [1., 2., 0.]
  >>> labels = range(1,len(data11)+1)
  >>> fo11 = FieldOutput(labels = labels, data=data11,position='node')
  >>> fo22 = FieldOutput(labels = labels, data=data22,position='node')
  >>> fo12 = FieldOutput(labels = labels, data=data12,position='node')
  >>> tensor = TensorFieldOutput(data11 = fo11, data22 = fo22, data12 = fo12 )
  >>> tensor2 = TensorFieldOutput(data11= fo22 )
  >>> tensor
  <TensorFieldOutput instance: 3 locations>
  >>> print tensor
  TensorFieldOutput instance
  Position = node
  Label	Data11	Data22	Data33	Data12	Data13	Data23
  1	1.0e+00	2.0e+00	0.0e+00	1.0e+00	0.0e+00	0.0e+00
  2	1.0e+00	4.0e+00	0.0e+00	2.0e+00	0.0e+00	0.0e+00
  3	1.0e+00	-1.0e+00	0.0e+00	0.0e+00	0.0e+00	0.0e+00
  >>> print tensor[1,2] 
  TensorFieldOutput instance
  Position = node
  Label	Data11	Data22	Data33	Data12	Data13	Data23
  1	1.0e+00	2.0e+00	0.0e+00	1.0e+00	0.0e+00	0.0e+00
  2	1.0e+00	4.0e+00	0.0e+00	2.0e+00	0.0e+00	0.0e+00
  >>> print tensor *2. + 1.
  TensorFieldOutput instance
  Position = node
  Label	Data11	Data22	Data33	Data12	Data13	Data23
  1	3.0e+00	5.0e+00	1.0e+00	3.0e+00	1.0e+00	1.0e+00
  2	3.0e+00	9.0e+00	1.0e+00	5.0e+00	1.0e+00	1.0e+00
  3	3.0e+00	-1.0e+00	1.0e+00	1.0e+00	1.0e+00	1.0e+00
  >>> print tensor ** 2 # Piecewise power
  TensorFieldOutput instance
  Position = node
  Label	Data11	Data22	Data33	Data12	Data13	Data23
  1	1.0e+00	4.0e+00	0.0e+00	1.0e+00	0.0e+00	0.0e+00
  2	1.0e+00	1.6e+01	0.0e+00	4.0e+00	0.0e+00	0.0e+00
  3	1.0e+00	1.0e+00	0.0e+00	0.0e+00	0.0e+00	0.0e+00
  >>> vector = VectorFieldOutput(data1 = fo11)
  >>> print tensor * vector # Matrix product
  VectorFieldOutput instance
  Position = node
  Label	Data1	Data2	Data3
  1	1.0	1.0	0.0
  2	1.0	2.0	0.0
  3	1.0	0.0	0.0
  >>> print tensor * tensor2 # Contracted tensor product
  FieldOutput instance
  Position = node
  Label	Data
  1	2.0
  2	4.0
  3	-1.0
  '''
  def __init__(self,data11=None,data22=None,data33=None, data12=None, data13=None, data23=None , position = 'node', dti='I', dtf = 'f'):
    from array import array
    self.position = position
    self.dti = dti
    self.dtf = dtf
    self.labels = array(dti,[])
    self.data11, self.data22, self.data33 = array(dtf,[]), array(dtf,[]), array(dtf,[])
    self.data12, self.data13, self.data23 = array(dtf,[]), array(dtf,[]), array(dtf,[])
    data = [data11, data22, data33, data12, data13, data23]
    isNone = [data11 == None, data22 == None, data33 == None, data12 == None, data13 == None, data23 == None]
    for d in data:
      if (isinstance(d,FieldOutput) == False) and (d != None):
        raise Exception, 'data11, data22, data33, data12, data13 and data23 must be FieldOutput instances.'
    if isNone != [True, True, True, True, True, True]:
      for i in [5,4,3,2,1,0]:
        d = data[i]
        if d != None: refData = d
      labels = refData.labels
      useShortInts = True
      positions = []
      for i in xrange(6):
        if data[i] == None: data[i] = ZeroFieldOutput_like(refData)
        if data[i].labels != labels:
          raise Exception, 'data11, data22, data33, data12, data13 and data23 must be fieldOutputs sharing the same labels.'    
        if data[i].dtf == 'd': self.dtf == 'd'
        if data[i].dti == 'I': useShortInts = False
        positions.append(data[i].position)
      if useShortInts: self.dti = 'H'            
      if len(set(positions)) > 1:
        raise Exception, 'inputs must have the same position or be None.'
      '''
      for i in xrange(len(labels)):
        l = labels[i]
        d1, d2, d3 = data[0].data[i] , data[1].data[i] , data[2].data[i]
        self.add_data(label = l, data1 = d1, data2 = d2, data3 = d3)
      '''
      self.labels = labels
      self.data11, self.data22, self.data33 = data[0].data, data[1].data, data[2].data  
      self.data12, self.data13, self.data23 = data[3].data, data[4].data, data[5].data  
  
  def get_component(self,number):
    '''
    Returns a component of the vector as a FieldOutput.
       
    :param number: requested coordinate number, 1 is x and so on.
    :type number: 11, 22, 33, 12, 13 or 23  
    :rtype: FieldOutput instance
    
    >>> v1 = Vec.get_coord(1)
    '''
    import numpy as np
    #from copy import copy
    if number not in [11, 22, 33, 12, 13, 23]:
      raise Exception, 'number must be 11, 22, 33, 12, 13 or 23'  
    dti, dtf = self.dti, self.dtf
    position = self.position
    if number == 11: data = self.data11
    if number == 22: data = self.data22
    if number == 33: data = self.data33
    if number == 12: data = self.data12
    if number == 13: data = self.data23
    if number == 23: data = self.data23
    return FieldOutput(position = position, data = data, labels=self.labels, dti=dti, dtf=dtf )
    
  def get_data(self,label):
    '''
    Returns the components (*11, 22, 33, 12, 13 or 23*) at a location with given label. 
    
    :param label: location's label.
    :type label: int > 0
    :rtype: float, float, float, float, float, float
        
    .. note:: Requesting data at a label that does not exist in the instance will just lead in a warning but if label is negative or is not int, then an Exception will be raised.
    '''
     
    if type(label) not in [int, long] or label <= 0:
      raise Exception, 'label must be int > 0, got {0}'.format(label)  
    if label in self.labels:
      i = self.labels.index(label)
      return self.data11[i], self.data22[i], self.data33[i], self.data12[i], self.data13[i], self.data23[i] 
    else:
      print 'Info: requesting data at non existant location, returning None'
      
  def add_data(self,label, data11=0., data22=0., data33=0., data12=0., data13=0., data23=0.):
    '''
    Adds one point to a VectorFieldOutput instance. Label must not already exist in the current FieldOutput, if not so, nothing will be changed. Data and label will be inserted in self.data, self.labels in order to keep self.labels sorted.
   
    :param label: labels of the nodes/elements where the field is evaluated.
    :type labels: int > 0
    :param data11: value of the component 11 of the field where it is evaluated.
    :type data1: float
    :param data22: value of the component 22 of the field where it is evaluated.
    :type data2: float
    :param data33: value of the component 33 of the field where it is evaluated.
    :type data3: float
    :param data12: value of the component 12 of the field where it is evaluated.
    :type data1: float
    :param data13: value of the component 13 of the field where it is evaluated.
    :type data2: float
    :param data23: value of the component 23 of the field where it is evaluated.
    :type data3: float
    '''
       
    from array import array
    dti,dtf = self.dti, self.dtf
    if label in self.labels:
      print 'data already exists at this node'
      return 
    else:
      self_data11, self_data22, self_data33, self_labels = self.data11, self.data22, self.data33, self.labels
      self_data12, self_data13, self_data23 = self.data12, self.data13, self.data23
      self_data11.append(data11)
      self_data22.append(data22)
      self_data33.append(data33)
      self_data12.append(data12)
      self_data13.append(data13)
      self_data23.append(data23)
      self_labels.append(label)
      zipped = zip(self_labels,self_data11, self_data22, self_data33,self_data12, self_data13, self_data23)
      labels,data11, data22, data33,data12, data13, data23 = zip(*sorted(zipped))
      self.data11 = array(self.dtf,data11)
      self.data22 = array(self.dtf,data22)
      self.data33 = array(self.dtf,data33)
      self.data12 = array(self.dtf,data12)
      self.data13 = array(self.dtf,data13)
      self.data23 = array(self.dtf,data23)
      self.labels = array(self.dti,labels)   
  
  def dump2vtk(self,name='tensorFieldOutput', header = True):
    '''
    Converts the TensorFieldOutput instance to VTK format which can be directly read by Mayavi2 or Paraview. This method is very useful to quickly and efficiently plot 3D mesh and fields.
    
    :param name: name used for the field in the output.
    :type name: string
    :rtype: string
    
    
    '''
    d11, d22, d33 = self.data11, self.data22, self.data33
    d12, d13, d23 = self.data12, self.data13, self.data23
    ld = len(d11)
    out = ""
    if header:
      if self.position == 'node': dataType = 'POINT_DATA'
      if self.position == 'element': dataType = 'CELL_DATA' 
      out += "{0} {1}\n".format(dataType, ld)
    out += 'TENSORS {0} float\n'.format(name)
    pattern = '{0} {3} {4}\n{3} {1} {5}\n{4} {5} {2}\n\n'
    for i in xrange(ld):
      out += pattern.format(d11[i], d22[i], d33[i], d12[i], d13[i], d23[i])
    return out
    
  def __str__(self):
    labels, position = self.labels, self.position
    data11, data22, data33 =  self.data11, self.data22, self.data33
    data12, data13, data23 =  self.data12, self.data13, self.data23
    out = 'TensorFieldOutput instance\nPosition = {0}\nLabel\tData11\tData22\tData33\tData12\tData13\tData23\n'.format(position)
    pattern = '{0}\t{1:.1e}\t{2:.1e}\t{3:.1e}\t{4:.1e}\t{5:.1e}\t{6:.1e}\n'
    for i in xrange(len(labels)): out += pattern.format(labels[i], data11[i], data22[i], data33[i], data12[i], data13[i], data23[i])
    return out
    
  def __repr__(self):
    l = len(self.labels)
    return '<TensorFieldOutput instance: {0} locations>'.format(l)
  def __getitem__(self,s):
    from array import array
    from copy import deepcopy
    from numpy import nan
    labs = []
    if type(s) in [int, long]:
      labs = [s]
    if type(s) is slice:
      start = s.start
      stop  = s.stop
      step  = s.step
      labs = range(start,stop,step)
    if type(s) in [tuple,list,array]:  
      for a in s:
        if type(a) in [int, long]:labs.append(a)
       
    labels = self.labels
    dtf = self.dtf
    dti = self.dti
    data11, data22, data33 = self.data11, self.data22, self.data33
    data12, data13, data23 = self.data12, self.data13, self.data23
    position = self.position
    fo = TensorFieldOutput(position = position, dti = dti, dtf = dtf)
    for l in labs:    
      if l in labels:
        i = labels.index(l)
        fo.add_data(label = l, data11 = data11[i], data22 = data22[i], data33 = data33[i], data12 = data12[i], data13 = data13[i], data23 = data23[i] )
      else:
        fo.add_data(label = l, data11 = nan, data22 = nan, data33 = nan, data12 = nan, data13 = nan, data23 = nan )  
    return fo
    
    
  def _PreProcess(self, other):
    s11, s22, s33 = self.get_component(11), self.get_component(22), self.get_component(33)
    s12, s13, s23 = self.get_component(12), self.get_component(13), self.get_component(23)
    if isinstance(other, TensorFieldOutput):
      o11, o22, o33 = other.get_component(11), other.get_component(22), other.get_component(33)
      o12, o13, o23 = other.get_component(12), other.get_component(13), other.get_component(23)
      o = (o11, o22, o33, o12, o13, o23)
      otype = 'tensor'
    if isinstance(other, VectorFieldOutput):
      o1, o2, o3 = other.get_coord(1), other.get_coord(2), other.get_coord(3)
      o = (o1, o2, o3)
      otype = 'vector'
    if isinstance(other, FieldOutput):
      o = other
      otype = 'scalar'
    if type(other) in [float, int, long]:
      o = other * OneFieldOutput_like(self.get_component(11))
      otype = 'scalar'
    return s11, s22, s33, s12, s13, s23, otype, o
     
  def _TensorPostProcess(self, out11, out22, out33, out12, out13, out23):
    out = TensorFieldOutput(data11 = out11, data22 = out22, data33 = out33, data12 = out12, data13 = out13, data23 = out23)
    return out
  
  def _VectorPostProcess(self, out1, out2, out3):
    out = VectorFieldOutput(data1 = out1, data2 = out2, data3 = out3)
    return out
  
  def _ScalarPostProcess(self, out):
    out = out * OneFieldOutput_like(self.get_component(11))
    return out
  
  def __add__(self, other):
    s11, s22, s33, s12, s13, s23, otype, o = self._PreProcess(other)
    if otype == 'scalar':
      out11, out22, out33 = s11 + o, s22 + o, s33 + o
      out12, out13, out23 = s12 + o, s13 + o, s23 + o
      return self._TensorPostProcess(out11, out22, out33, out12, out13, out23)
    if otype == 'vector':
      raise Exception, 'Tensors and vectors cannot be added.'
    if otype == 'tensor':
      o11, o22, o33, o12, o13, o23 = o[0], o[1], o[2], o[3], o[4], o[5] 
      out11, out22, out33 = s11 + o11, s22 + o22, s33 + o33
      out12, out13, out23 = s12 + o12, s13 + o13, s23 + o23
      return self._TensorPostProcess(out11, out22, out33, out12, out13, out23)
    
  __radd__ = __add__
      
  def __mul__(self, other): 
    s11, s22, s33, s12, s13, s23, otype, o = self._PreProcess(other)
    if otype == 'scalar': # term wise product
      out11, out22, out33 = s11 * o, s22 * o, s33 * o
      out12, out13, out23 = s12 * o, s13 * o, s23 * o
      return self._TensorPostProcess(out11, out22, out33, out12, out13, out23)
    if otype == 'vector': 
      o1, o2, o3 = o[0], o[1], o[2]
      out1 = s11 * o1 + s12 * o2 + s13 * o3   
      out2 = s12 * o1 + s22 * o2 + s23 * o3
      out3 = s13 * o1 + s23 * o2 + s33 * o3
      return self._VectorPostProcess(out1, out2, out3)
    if otype == 'tensor': # doubly contracted product
      o11, o22, o33, o12, o13, o23 = o[0], o[1], o[2], o[3], o[4], o[5] 
      out = s11 * o11 + s22 * o22 + s33 * o33 + 2 *( s12 * o12 + s13 * o13 + s23 * o23 )
      return self._ScalarPostProcess(out)
    
  __rmul__ = __mul__ # Here we accept that vector * tensor = tensor * vector. This is theoretically dangerous but in this context, it cannot lead to wrong results. It can be further discussed if necessary.
  
  
    
  def __neg__(self):
    other = 1.
    s11, s22, s33, s12, s13, s23, otype, o = self._PreProcess(other)
    return self._TensorPostProcess(-s11, -s22, -s33, -s12, -s13, -s23)
    
      
  def __sub__(self, other):
    return self + (-other)   
  
  def __rsub__(self, other):
    return -self + other
    
  def __div__(self, other):
    s11, s22, s33, s12, s13, s23, otype, o = self._PreProcess(other)
    if otype == 'scalar': 
      out11, out22, out33 = s11 / o, s22 / o, s33 / o
      out12, out13, out23 = s12 / o, s13 / o, s23 / o
      return self._TensorPostProcess(out11, out22, out33, out12, out13, out23)
    if otype in ['tensor, vector']:
      raise Exception, 'tensor division is only defined with scalars.'
         
  def __rdiv__(self, other):
    return other * self**-1
    
    
  def __pow__(self, other): # Piecewise power
    s11, s22, s33, s12, s13, s23, otype, o = self._PreProcess(other)
    if otype == 'scalar': 
      out11, out22, out33 = s11 ** o, s22 ** o, s33 ** o
      out12, out13, out23 = s12 ** o, s13 ** o, s23 ** o
      return self._TensorPostProcess(out11, out22, out33, out12, out13, out23)
    if otype in ['tensor, vector']:
      raise Exception, 'tensor power is only defined with scalars.'
    
  def __abs__(self): 
    other = 1.
    s11, s22, s33, s12, s13, s23, otype, o = self._PreProcess(other)
    return self._TensorPostProcess(abs(s11), abs(s22), abs(s33), abs(s12), abs(s13), abs(s23))
    
    
  def sum(self):
    '''
    Returns the sum of all components of the tensor.
    
    :rtype: ``FieldOutput`` instance.
    '''
    other = 1.
    s11, s22, s33, s12, s13, s23, otype, o = self._PreProcess(other)
    return s11 + s22 + s33 + s12 + s13 + s23
  
  def trace(self):
    '''
    Returns the trace of the tensor: :math:`trace(T) = T_{11} + T_{22} + T_{33}`
    
    :rtype: ``FieldOutput`` instance.
    '''
    other = 1.
    s11, s22, s33, s12, s13, s23, otype, o = self._PreProcess(other)
    return s11 + s22 + s33
  
  def deviatoric(self):
    '''
    Returns the deviatoric part tensor: :math:`T_{d} = T - T_{s}`
    
    :rtype: ``TensorFieldOutput`` instance
    '''
    return self - self.spheric()
    
  def spheric(self):
    '''
    Returns the spheric part of the tensor: :math:`T_{s} = \\frac{1}{3} trace(T) I_3`
    
    :rtype: ``TensorFieldOutput`` instance
    '''
    return (self.trace() / 3.) * Identity_like(self)
  
  def i1(self):
    '''
    Returns the first invariant, is equivalent to trace. 
    
    :rtype: ``FieldOutput`` instance.
    '''
    return self.trace()
    
  def pressure(self):
    '''
    Returns the pressure. 
    
    :rtype: ``FieldOutput`` instance.
    '''
    return -self.trace()/3.
   
  def i2(self):
    '''
    Returns the second invariant of the tensor defined as: :math:`inv2(T) = trace(T.T)`
        
    :rtype: ``FieldOutput`` instance.
    
    .. note:: this definition is the most practical one for mechanical engineering but not the only one possible. 
    '''
    return self*self
    
  def j2(self):
    '''
    Returns the second invariant of the deviatoric part of the tensor defined as: :math:`inv2(T) = trace(T_d.T_d)`
        
    :rtype: ``FieldOutput`` instance.
    
    .. note:: this definition is not the mathematical definition but is the most practical one for mechanical engineering. This should be debated.
    '''
    return self.deviatoric().i2()
  
  def i3(self):
    '''
    Returns the third invariant of the tensor: :math:`inv3(T) = \\det(T)`
    
    :rtype: ``FieldOutput`` instance.
    '''
    other = 1.
    s11, s22, s33, s12, s13, s23, otype, o = self._PreProcess(other)
    return (s11 * s22 * s33 + 2 * s12 * s23 * s13 - s12**2 * s33 - s23**2 * s11 - s13**2 * s22)
  
  def j3(self):
    '''
    Returns the third invariant of the deviatoric part of the tensor: :math:`inv3(T) = \\det(T_d)`
    
    :rtype: ``FieldOutput`` instance.
    '''
    return self.deviatoric().i3()
  
  def vonmises(self):
    '''
    Returns the von Mises equivalent equivalent *stress* of the tensor: :math:`vonmises(T) = \\sqrt{\\frac{3}{2} trace(T_d.T_d)}`
    
    :rtype: ``FieldOutput`` instance.
    '''
    return (1.5 * self.j2())**.5
    
  def tresca(self):
    '''
    Returns the tresca equivalent *stress* of the tensor: :math:`tresca(T) = max\\left( |t_1 - t_2|, |t_1 - t_3|, |t_2 - t_3|  \\right)` where :math:`t_i` is the i-est eigen value of T.
    
    :rtype: ``FieldOutput`` instance.
    '''
    s1, s2, s3, v1, v2, v3 = self.eigen()
    labels = self.labels
    pos = self.position
    dti, dtf = self.dti, self.dtf
    d1, d2, d3 = s1.data, s2.data, s3.data
    t = [] 
    for i in xrange(len(labels)):
      t.append( max([ abs(d1[i] - d2[i]), abs(d1[i] - d3[i]), abs(d3[i] - d2[i]) ]) )
    return FieldOutput(labels = labels, data = t, position = pos, dti = dti, dtf = dtf)
    
  def eigen(self):
    '''
    Returns the three eigenvalues with decreasing sorting and the 3 normed respective eigenvectors.
    
    :rtype: 3 ``FieldOutput`` instances and 3 ``VectorFieldOutput`` instances.
    
    >>> from abapy.postproc import FieldOutput, TensorFieldOutput, VectorFieldOutput, Identity_like
    >>> data11 = [0., 0., 1.]
    >>> data22 = [0., 0., -1]
    >>> data12 = [1., 2., 0.]
    >>> labels = range(1,len(data11)+1)
    >>> fo11 = FieldOutput(labels = labels, data=data11,position='node')
    >>> fo22 = FieldOutput(labels = labels, data=data22,position='node')
    >>> fo12 = FieldOutput(labels = labels, data=data12,position='node')
    >>> tensor = TensorFieldOutput(data11 = fo11, data22 = fo22, data12 = fo12 )
    >>> t1, t2, t3, v1, v2, v3 = tensor.eigen()
    >>> print t1
    FieldOutput instance
    Position = node
    Label	Data
    1	1.0
    2	2.0
    3	1.0
    >>> print v1
    VectorFieldOutput instance
    Position = node
    Label	Data1	Data2	Data3
    1	0.707106769085	0.707106769085	0.0
    2	0.707106769085	0.707106769085	0.0
    3	1.0	0.0	0.0
    '''
    
    from numpy.linalg import eig # eigen value/vectors function
    from numpy import array, zeros_like, float32, float64
    labels = self.labels
    pos = self.position
    dti = self.dti
    dtf = self.dtf
    if dtf == 'f': ndtf = float32
    if dtf == 'd': ndtf = float64
    eigval1, eigval2, eigval3 = [], [], [] 
    eigvec1, eigvec2, eigvec3 = [], [], []
    for i in xrange(len(labels)):
      label = labels[i]
      s11, s22, s33, s12, s13, s23 = self.get_data(label)
      t = array([ [s11, s12, s13] , [s12, s22, s23] , [s13, s23, s33] ], dtype = ndtf)
      vals, vects = eig(t)
      vects = vects.transpose() # so vects[i] is the eigenvector associated with vals[i]
      zipped = zip(vals, [0,1,2])
      vals, vec_pos = zip( *sorted(zipped, reverse=True) )
      eigval1.append(vals[0])
      eigval2.append(vals[1])
      eigval3.append(vals[2])
      eigvec1.append(vects[vec_pos[0]])
      eigvec2.append(vects[vec_pos[1]])
      eigvec3.append(vects[vec_pos[2]])
    eigvec1 = array(eigvec1)
    eigvec2 = array(eigvec2)
    eigvec3 = array(eigvec3)
    s1 = FieldOutput(position = pos, labels = labels, data = eigval1, dti = dti, dtf = dtf)
    s2 = FieldOutput(position = pos, labels = labels, data = eigval2, dti = dti, dtf = dtf)
    s3 = FieldOutput(position = pos, labels = labels, data = eigval3, dti = dti, dtf = dtf)
    v11 = FieldOutput(position = pos, labels = labels, data = eigvec1[:,0], dti = dti, dtf = dtf)
    v12 = FieldOutput(position = pos, labels = labels, data = eigvec1[:,1], dti = dti, dtf = dtf)
    v13 = FieldOutput(position = pos, labels = labels, data = eigvec1[:,2], dti = dti, dtf = dtf)
    v21 = FieldOutput(position = pos, labels = labels, data = eigvec2[:,0], dti = dti, dtf = dtf)
    v22 = FieldOutput(position = pos, labels = labels, data = eigvec2[:,1], dti = dti, dtf = dtf)
    v23 = FieldOutput(position = pos, labels = labels, data = eigvec2[:,2], dti = dti, dtf = dtf)
    v31 = FieldOutput(position = pos, labels = labels, data = eigvec3[:,0], dti = dti, dtf = dtf)
    v32 = FieldOutput(position = pos, labels = labels, data = eigvec3[:,1], dti = dti, dtf = dtf)
    v33 = FieldOutput(position = pos, labels = labels, data = eigvec3[:,2], dti = dti, dtf = dtf)
    v1 = VectorFieldOutput(data1 = v11, data2 = v12, data3 = v13)
    v2 = VectorFieldOutput(data1 = v21, data2 = v22, data3 = v23)
    v3 = VectorFieldOutput(data1 = v31, data2 = v32, data3 = v33)
    return s1, s2, s3, v1, v2, v3
    
def Identity_like(fo):
  '''
  A TensorFieldOutput containing only identity but with the same position, labels and dtypes as the input. 
 
  :param fo: tensor field output to be used.
  :type fo: TensorFieldOutput instance
  :rtype: TensorFieldOutput instance
  
  >>> from abapy.postproc import FieldOutput, TensorFieldOutput, Identity_like
  >>> data1 = [1,2,3,5,6,]
  >>> data2 = [1. for i in data1]
  >>> labels = range(1,len(data1)+1)
  >>> fo1, fo2 = FieldOutput(labels = labels, data=data1, position='node' ), FieldOutput(labels = labels, data=data2,position='node')
  >>> tensor = TensorFieldOutput(data11 = fo1, data22 = fo2 )
  >>> identity = Identity_like(tensor)
  >>> print identity
  TensorFieldOutput instance
  Position = node
  Label	Data11	Data22	Data33	Data12	Data13	Data23
  1	1.0e+00	1.0e+00	1.0e+00	0.0e+00	0.0e+00	0.0e+00
  2	1.0e+00	1.0e+00	1.0e+00	0.0e+00	0.0e+00	0.0e+00
  3	1.0e+00	1.0e+00	1.0e+00	0.0e+00	0.0e+00	0.0e+00
  4	1.0e+00	1.0e+00	1.0e+00	0.0e+00	0.0e+00	0.0e+00
  5	1.0e+00	1.0e+00	1.0e+00	0.0e+00	0.0e+00	0.0e+00

  '''
  from copy import copy
  from array import array
  from numpy import ones_like
  if isinstance(fo,TensorFieldOutput) == False:
    raise Exception, 'input must be TensorFieldOutput instance.'
  d11 = OneFieldOutput_like(fo.get_component(11))
  d22 = OneFieldOutput_like(fo.get_component(11))
  d33 = OneFieldOutput_like(fo.get_component(11))
  return TensorFieldOutput(data11 = d11, data22 = d22, data33 = d33)
