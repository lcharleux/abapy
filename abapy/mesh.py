'''
Mesh
====
'''

import copy

class Nodes(object):
  '''
  Manages nodes for finite element modeling pre/postprocessing and further graphical representations.
  
  :param labels: list of node labels.  
  :type labels: list of int > 0
  :param x: x coordinate of nodes.  
  :type x: list floats
  :param y: y coordinate of nodes.  
  :type y: list floats
  :param z: z coordinate of nodes.  
  :type z: list floats
  :param sets: node sets  
  :type sets: dict with str keys and where values are list of ints > 0.
  :param dti: int data type in array.array
  :type dti: 'I' or 'H'
  :param dtf: float data type in array.array
  :type dtf: 'f' or 'd'
  
  
  >>> from abapy.mesh import Nodes 
  >>> labels = [1,2]
  >>> x = [0., 1.]
  >>> y = [0., 2.]
  >>> z = [0., 0.]
  >>> sets = {'mySet': [1,2]}
  >>> nodes = Nodes(labels = labels, x = x, y = y, z = z, sets = sets)
  >>> nodes
  <Nodes class instance: 2 nodes>
  >>> print nodes
  Nodes class instance:
  Nodes:
  Label	x	y	z
  1	0.0	0.0	0.0
  2	1.0	2.0	0.0
  Sets:
  Label	Nodes
  myset	1,2
  >>> from abapy.mesh import Nodes
  >>> labels = range(1,11) # 10 nodes
  >>> x = labels
  >>> y = labels
  >>> z = [0. for i in x]
  >>> nodes = Nodes(labels=labels, x=x, y=y, z=z)
  >>> nodes.add_set('myset',[4,5,6,9]) # A set
  >>> print nodes
  Nodes class instance:
  Nodes:
  Label	x	y	z
  1	1.0	1.0	0.0
  2	2.0	2.0	0.0
  3	3.0	3.0	0.0
  4	4.0	4.0	0.0
  5	5.0	5.0	0.0
  6	6.0	6.0	0.0
  7	7.0	7.0	0.0
  8	8.0	8.0	0.0
  9	9.0	9.0	0.0
  10	10.0	10.0	0.0
  Sets:
  Label	Nodes
  myset	4,5,6,9
  >>> print nodes[5] # requesting node 5
  Nodes class instance:
  Nodes:
  Label	x	y	z
  5	5.0	5.0	0.0
  Sets:
  Label	Nodes
  myset	5
  >>> print nodes[5,4,10]  # requesting nodes 5, 4 and 10. Note that output has ordered nodes and kept sets.
  Nodes class instance:
  Nodes:
  Label	x	y	z
  4	4.0	4.0	0.0
  5	5.0	5.0	0.0
  10	10.0	10.0	0.0
  Sets:
  Label	Nodes
  myset	5,4
  >>> print nodes['myset'] # requesting nodes using set key
  Nodes class instance:
  Nodes:
  Label	x	y	z
  4	4.0	4.0	0.0
  5	5.0	5.0	0.0
  6	6.0	6.0	0.0
  9	9.0	9.0	0.0
  Sets:
  Label	Nodes
  myset	4,5,6,9
  >>> print nodes['myset',10] # mixed request: nodes in myset AND node 10.
  Nodes class instance:
  Nodes:
  Label	x	y	z
  4	4.0	4.0	0.0
  5	5.0	5.0	0.0
  6	6.0	6.0	0.0
  9	9.0	9.0	0.0
  10	10.0	10.0	0.0
  Sets:
  Label	Nodes
  myset	4,5,6,9
  >>> print nodes[1:9:2] # slice
  Nodes class instance:
  Nodes:
  Label	x	y	z
  1	1.0	1.0	0.0
  3	3.0	3.0	0.0
  5	5.0	5.0	0.0
  7	7.0	7.0	0.0
  Sets:
  Label	Nodes
  myset	5

  '''
  def __init__(self,labels=[],x=[],y=[],z=[],sets={},dtf='f',dti='I'):
    from array import array
    from copy import deepcopy
    self.dtf = dtf
    self.dti = dti
    if type(labels) == str: labels = [labels]
    if type(x) in [float, int] : x = [x]
    if type(y) in [float, int] : y = [x]
    if type(z) in [float, int] : z = [x]
    if len(set([len(labels),len(x),len(y),len(z)])) != 1: 
      raise ValueError('labels, x, y and z args should have the same length')
    
    self.x = array(dtf,[])
    self.y = array(dtf,[])
    self.z = array(dtf,[])
    self.labels = array(dti,[])
    self.sets = {}
    for i in xrange(len(labels)):
      self.add_node(label = labels[i], x = x[i], y = y[i], z = z[i])
    
    for k in sets.keys():
      self.add_set(k, sets[k])
  def add_node(self,label=None,x=0.,y=0.,z=0.,toset=None):
    '''
    Adds one node to Nodes instance.
    
    :param label: If None, label is automatically chosen to be the highest existing label + 1 (default: None). If label (and susequently node) already exists, a warning is printed and the node is not added and sets that could be created ar not created.
    :type labels: int > 0
    :param x: x coordinate of node.  
    :type x: float
    :param y: y coordinate of node.  
    :type y: float
    :param z: z coordinate of node.  
    :type z: float
    :param tosets: set(s) to which the node should be added. If a set does not exist, it is created. If None, the node is not added to any set.
    :type sets: None, string or list of strings.
    
    >>> from abapy.mesh import Nodes 
    >>> nodes = Nodes()
    >>> nodes.add_node(label = 10, x = 0., y = 0., z = 0., toset = 'firstSet')
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    10	0.0	0.0	0.0
    Sets:
    Label	Nodes
    firstset	10
    >>> nodes.add_node(x = 0., y = 0., z = 0., toset = ['firstSet', 'secondSet'])
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    10	0.0	0.0	0.0
    11	0.0	0.0	0.0
    Sets:
    Label	Nodes
    firstset	10,11
    secondset	11

    '''
    from copy import deepcopy
    labels = self.labels
    sets = self.sets
    X, Y, Z = self.x, self.y, self.z
    if len(labels) == 0:
      if label == None: label = 1
      if label < 1: raise Exception, 'node labels must be > 0.'
      labels.append(label)
      X.append(x)
      Y.append(y)
      Z.append(z)
    else:
      if label == None: 
        label = max(labels)+1
        labels.append(label)
        X.append(x)
        Y.append(y)
        Z.append(z)
      else:
        if label not in labels:
          if label > max(labels):
            labels.append(label)
            X.append(x)
            Y.append(y)
            Z.append(z)
          else:
            dummy = deepcopy(labels)
            dummy.append(label)
            i = sorted(dummy).index(label)
            del dummy # Is it necessary ?
            labels.insert(i,label)
            X.insert(i,x)
            Y.insert(i,y)
            Z.insert(i,z)
        else:
          print 'Info: node with label {0} already exists, nothing changed.'.format(label)
          return
    if toset != None:
      if type(toset) is str:
        self.add_set(toset,[label])
      else:
        if '__getitem__' in dir(toset):
          for s in toset:
            self.add_set(s,[label])
  def drop_node(self,label):
    '''
    Removes one node to Nodes instance. The node is also removed from sets, if a set happens to be empty, it is also removed.
    
    :param label: node be removed's label.
    :type label: int > 0
    
    >>> from abapy.mesh import Nodes 
    >>> nodes = labels = [1,2]
    >>> x = [0., 1.]
    >>> y = [0., 2.]
    >>> z = [0., 0.]
    >>> sets = {'mySet': [2]}
    >>> nodes = Nodes(labels = labels, x = x, y = y, z = z, sets = sets)
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    1	0.0	0.0	0.0
    2	1.0	2.0	0.0
    Sets:
    Label	Nodes
    myset	2
    >>> nodes.drop_node(2)
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    1	0.0	0.0	0.0
    Sets:
    Label	Nodes'''
    
    labels = self.labels
    x,y,z = self.x, self.y, self.z 
    sets = self.sets
    i = labels.index(label)
    labels.remove(label)
    x.pop(i)
    y.pop(i)
    z.pop(i)
    for k in sets.keys():
      if label in sets[k]: sets[k].remove(label)  
      if len(sets[k]) == 0: del sets[k]
  def add_set(self,label,nodes):
    '''
    Adds a node set to the Nodes instance or appends nodes to existing node sets. 
    
    :param label: set to be added's label.
    :type label: string
    :param nodes: nodes to be added in the set.
    :type nodes: int or list of ints
    
    .. note:: set labels are always lower case in this class to be case insensitive. This way to proceed is coherent with Abaqus.
    
    >>> from abapy.mesh import Nodes
    >>> nodes = Nodes()
    >>> labels = [1,2]
    >>> x = [0., 1.]
    >>> y = [0., 2.]
    >>> z = [0., 0.]
    >>> sets = {'mySet': 2}
    >>> nodes = Nodes(labels = labels, x = x, y = y, z = z, sets = sets)
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    1	0.0	0.0	0.0
    2	1.0	2.0	0.0
    Sets:
    Label	Nodes
    myset	2
    >>> nodes.add_set('MYSET',1)
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    1	0.0	0.0	0.0
    2	1.0	2.0	0.0
    Sets:
    Label	Nodes
    myset	2,1
    >>> nodes.add_set('MyNeWseT',[1,2])
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    1	0.0	0.0	0.0
    2	1.0	2.0	0.0
    Sets:
    Label	Nodes
    myset	2,1
    mynewset	1,2'''
    
    from array import array
    sets = self.sets
    label2 = label.lower()
    '''
    if label != label2:
      try: 
        print 'Info: node set label {0} was changed to {1}.'.format(label, label2)
      except:
        print 'A node set was renamed'
    '''
    label = label2
    if label not in sets: sets[label] = array(self.dti,[])
    s = sets[label]
    if type(nodes) == int: nodes = [nodes]
    if '__getitem__' in dir(nodes):
      for node in nodes:
        if node in self.labels:
          if node not in self.sets[label]: 
            s.append(node)
          else:
           print 'Info: node {0} was already in set {1}, node was not added to set'.format(node, label)
        else:
          print 'Info: node {0} does not exist, node was not added to set'.format(node)
  
  def add_set_by_func(self, name, func):
    '''
    Creates a node set using a function of x, y, z and labels (given as ``numpy.array``). Must get back a boolean array of the same size.
    
    :param name: set label.
    :type name: string
    :param func: function of x, y ,z and labels
    :type func: function
     
    >>> mesh.nodes.add_set_by_func('setlabel', lambda x, y, z, labels: x == 0.)

     '''
    from numpy import array, float32, float64, uint32, where
    if type(name) is not str: raise Exception, 'set labels must be strings, got {0}.'.format(type(name))
    if self.dtf == 'f': n_dtf = float32
    if self.dtf == 'd': n_dtf = float64
    x = array(self.x, dtype = n_dtf)
    y = array(self.y, dtype = n_dtf)
    z = array(self.z, dtype = n_dtf)
    labels = array(self.labels, dtype = uint32)
    out = func(x, y, z, labels)
    out = where(out == True)[0].tolist()
    labs = labels[out]
    if labels != []: self.add_set(name, labs)
    
  def add_set_by_func_2D(self, name, func):
    '''
    Creates a node set using a function of x, y and labels (given as ``numpy.array``). Must get back a boolean array of the same size.
    
    :param name: set label.
    :type name: string
    :param func: function of x, y and labels
    :type func: function
     
    >>> mesh.nodes.add_set_by_func('setlabel', lambda x, y, labels: x == 0.)

     '''
    from numpy import array, float32, float64, uint32, where
    if type(name) is not str: raise Exception, 'set labels must be strings, got {0}.'.format(type(name))
    if self.dtf == 'f': n_dtf = float32
    if self.dtf == 'd': n_dtf = float64
    x = array(self.x, dtype = n_dtf)
    y = array(self.y, dtype = n_dtf)
    labels = array(self.labels, dtype = uint32)
    out = func(x, y, labels)
    out = where(out == True)[0].tolist()
    labs = labels[out]
    if labels != []: self.add_set(name, labs)  
  
  def translate(self,x=0.,y=0.,z=0.):
    '''
    Translates all the nodes.
    
    :param x: translation along x value.
    :type x: float
    :param y: translation along y value.
    :type y: float
    :param z: translation along z value.
    :type z: float
    
    >>> from abapy.mesh import Nodes
    >>> nodes = Nodes()
    >>> labels = [1,2]
    >>> x = [0., 1.]
    >>> y = [0., 2.]
    >>> z = [0., 0.]
    >>> sets = {'mySet': 2}
    >>> nodes = Nodes(labels = labels, x = x, y = y, z = z, sets = sets)
    >>> nodes.translate(x = 1., y=0., z = -4.)
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    1	1.0	0.0	-4.0
    2	2.0	2.0	-4.0
    Sets:
    Label	Nodes
    myset	2'''
    from numpy import ones_like
    
    def func(xx, yy, zz, labels):
      return x*ones_like(xx), y*ones_like(yy), z*ones_like(zz)
    u = self.eval_vectorFunction(func)
    self.apply_displacement(u)
  
  def dump2inp(self):
    '''
    Dumps Nodes instance to Abaqus INP format.
    
    :rtype: string
    
    >>> from abapy.mesh import Nodes
    >>> nodes = Nodes()
    >>> labels = [1,2]
    >>> x = [0., 1.]
    >>> y = [0., 2.]
    >>> z = [0., 0.]
    >>> sets = {'mySet': 2}
    >>> nodes = Nodes(labels = labels, x = x, y = y, z = z, sets = sets)
    >>> out = nodes.dump2inp()''' 
    
    labels = self.labels
    x, y, z = self.x, self.y, self.z
    sets = self.sets
    out ='**----------------------------------\n** NODES\n**----------------------------------\n*NODE, NSET=ALLNODES\n'
    pattern = '  {0}, {1}, {2}, {3}\n'
    for i in xrange(len(labels)):
      out += pattern.format(labels[i],x[i],y[i],z[i])
    out +='**----------------------------------\n** NODE SETS\n**----------------------------------\n'
    pattern = '*NSET, NSET={0}\n'
    pattern1 = '{0},'
    for k in sets.keys():
      nset = sets[k]
      out += pattern.format(k.upper())
      nCount = 0
      for n in nset:
        out += pattern1.format(n)
        nCount += 1
        if nCount >= 9: 
          out += '\n'
          nCount = 0
      out += '\n' 
    return out
  def __repr__(self):
    pattern = '<Nodes class instance: {} nodes>'
    return pattern.format(len(self.labels))  
  def __str__(self):
    labels = self.labels
    x =  self.x
    y = self.y
    z = self.z
    out ='Nodes class instance:\nNodes:\nLabel\tx\ty\tz\n'
    for i in xrange(len(labels)):
      out +=  '{0}\t{1}\t{2}\t{3}\n'.format(labels[i],x[i],y[i],z[i])
    out+= 'Sets:\nLabel\tNodes\n'
    for k in self.sets.keys():
      out +=  '{0}\t'.format(k)
      for n in self.sets[k]:
        out += '{0},'.format(n)
      out = out[0:-1] +'\n'
    return out
  def __getitem__(self,s):
    from array import array
    from copy import deepcopy
    if type(s) in [int, long]:
      labs = [s]
    if type(s) is slice:
      start = s.start
      stop  = s.stop
      step  = s.step
      labs = range(start,stop,step)
    if type(s) in [tuple,list]:  
      labs = []
      for a in s:
       if type(a) in [int, long]:labs.append(a)
       if type(a) is str:labs += self.sets[a.lower()].tolist()
    if type(s) is str:
      s = s.lower() 
      if s in self.sets.keys():
        labs = self.sets[s]
      else:
        labs = []
    labels = self.labels
    dtf = self.dtf
    dti = self.dti
    sets = self.sets
    x = self.x
    y = self.y
    z = self.z
    N = Nodes()
    for l in labs:
      toset = []
      i = labels.index(l)
      for sk in sets.keys():
        if l in sets[sk]:
          toset.append(sk)
      N.add_node(label = l, x = x[i], y = y[i], z = z[i], toset = toset)
    return N      
  def drop_set(self,label):
    '''
    Drops a set without removing elements and nodes.
    
    :param label: label of the to be removed.
    :type label: string
    
    
    >>> from abapy.mesh import Nodes
    >>> labels = range(1,11)
    >>> x = labels
    >>> y = labels
    >>> z = [0. for i in x]
    >>> nodes = Nodes(labels=labels, x=x, y=y, z=z)
    >>> nodes.add_set('myset',[4,5,6,9])
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    1	1.0	1.0	0.0
    2	2.0	2.0	0.0
    3	3.0	3.0	0.0
    4	4.0	4.0	0.0
    5	5.0	5.0	0.0
    6	6.0	6.0	0.0
    7	7.0	7.0	0.0
    8	8.0	8.0	0.0
    9	9.0	9.0	0.0
    10	10.0	10.0	0.0
    Sets:
    Label	Nodes
    myset	4,5,6,9
    >>> nodes.drop_set('someSet')
    Info: sets someset does not exist and cannot be dropped.
    >>> nodes.drop_set('MYSET')
    >>> print nodes
    Nodes class instance:
    Nodes:
    Label	x	y	z
    1	1.0	1.0	0.0
    2	2.0	2.0	0.0
    3	3.0	3.0	0.0
    4	4.0	4.0	0.0
    5	5.0	5.0	0.0
    6	6.0	6.0	0.0
    7	7.0	7.0	0.0
    8	8.0	8.0	0.0
    9	9.0	9.0	0.0
    10	10.0	10.0	0.0
    Sets:
    Label	Nodes'''
    
    label = label.lower()
    if label in self.sets.keys():
      del self.sets[label]
    else:
      print 'Info: sets {0} does not exist and cannot be dropped.'.format(label)     
  
  def eval_function(self, function):
    '''
    Evals a function at each node and returns a ``FieldOutput`` instance.
    
    :param function: a function with arguments x, y and z (float ``numpy.arrays`` containing nodes coordinates) and labels (int ``numpy.array``). Field should not depend on labels but on some vicious problem, it could be useful. The function should return 1 array.
    :type function: function
    :rtype: ```FieldOutput``` instance.
    
    .. plot:: example_code/mesh/Nodes-eval_function.py
       :include-source:
    '''
    from postproc import FieldOutput 
    from numpy import array as n_array
    from numpy import float32, float64, uint32, uint64
    dtf, dti = self.dtf, self.dti
    if dtf == 'f': n_dtf = float32
    if dtf == 'd': n_dtf = float64
    if dti == 'H': n_dti = float16
    if dti == 'I': n_dti = float32
    x = n_array(self.x, dtype = n_dtf)
    y = n_array(self.y, dtype = n_dtf)
    z = n_array(self.z, dtype = n_dtf)
    labels = self.labels
    n_labels = n_array(labels, dtype = n_dti) 
    data = function(x, y, z, n_labels)
    return FieldOutput(labels = labels, data = data, dti = dti, dtf = dtf)
    
  def eval_vectorFunction(self, function):
    '''
    Evals a vector function at each node and returns a ``VectorFieldOutput`` instance.
    
    :param function: a vector function with arguments x, y and z (float ``numpy.arrays`` containing nodes coordinates) and labels (int ``numpy.array``). Field should not depend on labels but on some vicious problem, it could be useful. The function should return 3 arrays.
    :type function: function
    :rtype: ```FieldOutput``` instance.
    
    .. plot:: example_code/mesh/Nodes-eval_vectorFunction.py
       :include-source:
    '''
    from postproc import VectorFieldOutput, FieldOutput
    from numpy import array as n_array
    from numpy import float32, float64, uint32, uint64
    dtf, dti = self.dtf, self.dti
    if dtf == 'f': n_dtf = float32
    if dtf == 'd': n_dtf = float64
    if dti == 'H': n_dti = float16
    if dti == 'I': n_dti = float32
    x = n_array(self.x, dtype = n_dtf)
    y = n_array(self.y, dtype = n_dtf)
    z = n_array(self.z, dtype = n_dtf)
    labels = self.labels
    n_labels = n_array(labels, dtype = n_dti) 
    data1, data2, data3 = function(x, y, z, n_labels)
    data1 = FieldOutput(labels = labels, data = data1, dti = dti, dtf = dtf)
    data2 = FieldOutput(labels = labels, data = data2, dti = dti, dtf = dtf)
    data3 = FieldOutput(labels = labels, data = data3, dti = dti, dtf = dtf)
    return VectorFieldOutput(data1 = data1, data2=data2, data3=data3)
    
  def eval_tensorFunction(self, function):
    '''
    Evaluates a tensor function at each node and returns a ``tensorFieldOutput`` instance.
    
    :param function: a tensor function with arguments x, y and z (float ``numpy.arrays`` containing nodes coordinates) and labels (int ``numpy.array``). Field should not depend on labels but on some vicious problem, it could be useful. The function should return 6 arrays corresponding to indices ordered as follows: 11, 22, 33, 12, 13, 23.
    :type function: function
    :rtype: ```FieldOutput``` instance.
    
    .. plot:: example_code/mesh/Nodes-eval_tensorFunction.py
       :include-source:
    '''
    from postproc import TensorFieldOutput, FieldOutput
    from numpy import array as n_array
    from numpy import float32, float64, uint32, uint64
    dtf, dti = self.dtf, self.dti
    if dtf == 'f': n_dtf = float32
    if dtf == 'd': n_dtf = float64
    if dti == 'H': n_dti = float16
    if dti == 'I': n_dti = float32
    x = n_array(self.x, dtype = n_dtf)
    y = n_array(self.y, dtype = n_dtf)
    z = n_array(self.z, dtype = n_dtf)
    labels = self.labels
    n_labels = n_array(labels, dtype = n_dti) 
    data11, data22, data33, data12, data13, data23 = function(x, y, z, n_labels)
    data11 = FieldOutput(labels = labels, data = data11, dti = dti, dtf = dtf)
    data22 = FieldOutput(labels = labels, data = data22, dti = dti, dtf = dtf)
    data33 = FieldOutput(labels = labels, data = data33, dti = dti, dtf = dtf)
    data12 = FieldOutput(labels = labels, data = data12, dti = dti, dtf = dtf)
    data13 = FieldOutput(labels = labels, data = data13, dti = dti, dtf = dtf)
    data23 = FieldOutput(labels = labels, data = data23, dti = dti, dtf = dtf)
    return TensorFieldOutput(data11 = data11, data22=data22, data33=data33, data12=data12, data13= data13, data23=data23)
    
  def apply_displacement(self,disp):
    '''
    Applies a displacement field to the nodes.
    
    :param disp: displacement field.
    :type disp: ``VectorFieldOutput`` instance
    
    .. plot:: example_code/mesh/Nodes-apply_displacement.py
       :include-source:
    '''
    from postproc import VectorFieldOutput
    from copy import deepcopy
    from numpy import array as n_array
    from array import array as a_array
    from numpy import float32, float64
    dtf = self.dtf
    if dtf == 'f': n_dtf = float32
    if dtf == 'd': n_dtf = float64
    if isinstance(disp,VectorFieldOutput) == False:
      raise Exception, 'disp must be VectorFieldOutput instance.'
    # labels = set(mesh.nodes.labels.tolist() + disp.labels.tolist()) # Dangerous method
    n_labels, u_labels = self.labels, disp.labels
    if n_labels != u_labels: raise Exception, 'Nodes labels and VectorFieldOutput labels must be identical.'
    n1, n2, n3 = n_array(self.x, dtype=n_dtf), n_array(self.y, dtype=n_dtf), n_array(self.z, dtype=n_dtf)
    u1, u2, u3 = n_array(disp.data1, dtype=n_dtf), n_array(disp.data2, dtype=n_dtf), n_array(disp.data3, dtype=n_dtf)
    self.x = a_array(dtf, ( n1 + u1 ).tolist() )
    self.y = a_array(dtf, ( n2 + u2 ).tolist() )
    self.z = a_array(dtf, ( n3 + u3 ).tolist() )
  
  def closest_node(self, label):
    '''
    Finds the closest node of an existing node.
    
    :param label: node label to be used.
    :type label: int > 0
    :rtype: label (int > 0) and distance (float) of the closest node.
    '''
    from numpy import array, float32, uint32, where, delete
    pos = self.labels.index(label)
    X = array(self.x, dtype = float32)
    Y = array(self.y, dtype = float32)
    Z = array(self.z, dtype = float32)
    labels = array(self.labels, dtype = uint32)
    X = delete(X,pos)
    Y = delete(Y,pos)
    Z = delete(Z,pos)
    labels = delete(labels,pos)
    x, y, z = self.x[pos], self.y[pos], self.z[pos]
    Dist = ( (X-x)**2 + (Y-y)**2 + (Z-z)**2 )**.5
    dist = Dist.min()
    i = where(Dist == dist)[0][0]
    lab = labels[i]
    return lab, dist    
  
  def replace_node(self, old, new):
    '''
    Replaces a node of given label (old) by another existing node (new).
    '''
    nlabels = self.labels
    nsets = self.sets
    nlabels = self.labels
    x, y, z = self.x, self.y, self.z
    # Some verifications
    if old not in nlabels or new not in nlabels: raise Exception, 'Both node labels must exist.'
    # Let's proceed
    i_old, i_new = nlabels.index(old), nlabels.index(new)
    nlabels.pop(i_old)
    x.pop(i_old)
    y.pop(i_old)
    z.pop(i_old)
    for nsk in nsets.keys():
      ns = nsets[nsk] 
      if old in ns:
        ns.remove(old)
        if new not in ns: ns.append(new)
  def boundingBox(self, margin = 0.1):
    '''
    Returns the dimensions of a cartesian box containing the mesh with a relative margin.
    
    :param margin: relative margin of the box. O. means no margin, 0.1 is default.
    :type margin: float 
    :rtype: tuple containing 3 tuples with x, y and z limits of the box.    
    '''
    xmin, xmax = min(self.x), max(self.x)
    ymin, ymax = min(self.y), max(self.y)
    zmin, zmax = min(self.z), max(self.z)
    dx, dy, dz = xmax - xmin, ymax - ymin,  zmax - zmin
    return ( (xmin - margin * dx, xmax + margin * dx ), (ymin - margin * dy, ymax + margin * dy ), (zmin - margin * dz, zmax + margin * dz ) )
  
  def apply_reflection(self,point = (0., 0., 0.), normal = (1., 0., 0.)):
    '''
    Applies a reflection symmetry to the nodes instance. The reflection plane is defined by a point and a normal direction.
    
    :param point: coordinates of a point of the reflection plane.
    :type point: tuple or list containing 3 floats
    :param normal: normal vector to the reflection plane
    :type normal: tuple or list containing 3 floats
    
    :rtype: ``Mesh`` instances
    '''
    from copy import copy
    from numpy import array as n_array
    from numpy import float32, float64, sqrt
    from array import array as a_array
    if len(point) != 3: raise Exception, 'point must be a list  of length 3.'
    if len(normal) != 3: raise Exception, 'normal must be a list  of length 3.'
    if self.dtf == 'f' : a_dtf = float32
    if self.dtf == 'd' : a_dtf = float64
    xp, yp, zp = point[0], point[1], point[2] 
    xn, yn, zn = normal[0], normal[1], normal[2] 
    norm = sqrt( xn**2 + yn**2 + zn**2 ) 
    xn, yn, zn = xn / norm, yn / norm, zn / norm 
    x, y, z = n_array(self.x, dtype= a_dtf), n_array(self.y, dtype= a_dtf), n_array(self.z, dtype= a_dtf)
    vx, vy, vz = x - xp, y - yp, z -zp
    vn = vx * xn + vy * yn + vz *zn
    vx, vy, vz = vx - 2 * vn * xn, vy - 2 * vn * yn, vz - 2 * vn * zn
    x, y, z = vx + xp, vy + yp, vz + zp
    out = copy(self)
    out.x, out.y, out.z = a_array(out.dtf, x.tolist()), a_array(out.dtf, y.tolist()), a_array(out.dtf, z.tolist())
    return out
    
    
class Mesh(object):
  '''
  Manages meshes for finite element modeling pre/postprocessing and further graphical representations.
  
  :param nodes: nodes container. If None, a void Nodes instance will be used. The values of dti and dtf used by nodes are extended to mesh.
  :type nodes: Nodes class instance or None
  :param labels: elements labels
  :type label: list of ints > 0
  :param connectivity: elements connectivities using node labels
  :type connectivity: list of lists each containing ints > 0
  :param space: elements embedded spaces. This formulation is simple and allows to distinguish 1D elements (space = 1), surface elements (space = 2) and volumic elements (space = 3)
  :type space: list of ints in [1,2,3]
  :param name: elements names used, for example in a FEM code: 'CAX4, C3D8, ...'
  :type name: list of strings
  :param sets: element sets
  :type sets: dict with string keys and list of ints > 0 values
  :type surfaces: element surfaces.
  :param surface: dict with str keys containing tuples with 2 elements, the first being the name of an element set and the second the number of the face.
  
  >>> from abapy.mesh import Mesh, Nodes
  >>> mesh = Mesh()
  >>> nodes = mesh.nodes
  >>> # Adding some nodes
  >>> nodes.add_node(label = 1, x = 0. ,y = 0. , z = 0.)
  >>> nodes.add_node(label = 2, x = 1. ,y = 0. , z = 0.)
  >>> nodes.add_node(label = 3, x = 1. ,y = 1. , z = 0.)
  >>> nodes.add_node(label = 4, x = 0. ,y = 1. , z = 0.)
  >>> nodes.add_node(label = 5, x = 2. ,y = 0. , z = 0.)
  >>> nodes.add_node(label = 6, x = 2. ,y = 1. , z = 0.)
  >>> # Adding some elements
  >>> mesh.add_element(label=1, connectivity = (1,2,3,4), space =2, name = 'QUAD4', toset='mySet' )
  >>> mesh.add_element(label=2, connectivity = (2,5,6,3), space =2, name = 'QUAD4' )
  >>> print mesh[1]
  Mesh class instance:
  Elements:
  Label	Connectivity		Space	Name
  1	[1L, 2L, 3L, 4L]	2D	QUAD4
  Sets:
  Label	Elements
  myset	1
  >>> print mesh[1,2] # requesting elements with labels 1 and 2
  Mesh class instance:
  Elements:
  Label	Connectivity		Space	Name
  1	[1L, 2L, 3L, 4L]	2D	QUAD4
  2	[2L, 5L, 6L, 3L]	2D	QUAD4
  Sets:
  Label	Elements
  myset	1
  >>> print mesh[1:2:1] # requesting elements with labels in range(1,2,1)
  Mesh class instance:
  Elements:
  Label	Connectivity		Space	Name
  1	[1L, 2L, 3L, 4L]	2D	QUAD4
  Sets:
  Label	Elements
  myset	1
  >>> print mesh['mySet']
  Mesh class instance:
  Elements:
  Label	Connectivity		Space	Name
  1	[1L, 2L, 3L, 4L]	2D	QUAD4
  Sets:
  Label	Elements
  myset	1
  >>> print mesh['myset'] # requesting elements that belong to set 'myset'
  Mesh class instance:
  Elements:
  Label	Connectivity		Space	Name
  1	[1L, 2L, 3L, 4L]	2D	QUAD4
  Sets:
  Label	Elements
  myset	1
  >>> print mesh['ImNoSet']
  Mesh class instance:
  Elements:
  Label	Connectivity		Space	Name
  Sets:
  Label	Elements'''
  
  def __init__(self,nodes = None,connectivity=[],space=[],labels=[],name=None, sets={}, surfaces = {}):
    from array import array
    if nodes == None: nodes = Nodes() 
    if isinstance(nodes,Nodes):
      self.nodes = nodes 
      self.dti = nodes.dti
      self.dtf = nodes.dtf
      dti = self.dti
      dtf = self.dtf
    else:
      raise Exception, 'nodes argument must be a Nodes class instance.'
    if name == None: name = [None for i in labels]
    self.labels = array(dti,[])
    self.connectivity = []
    self.space = array('H',[])
    self.name = []
    self.sets = {}
    self.surfaces = {}
    self.fields = {}
    l = len(labels)
    if len(connectivity) != len(labels): raise Exception, 'connectivity must have the same length as labels.'
    if len(space) != len(labels): raise Exception, 'space must have the same length as labels.'
    if len(name) != len(labels): raise Exception, 'if not None, name must have the same length as labels.'
    for i in xrange(len(labels)):
      self.add_element(label = labels[i],connectivity = connectivity[i],space = space[i],name = name[i])
    if type(sets) is not dict: raise Exception, 'sets type must be dict.'
    for k in sets.keys():
      self.add_set(k, sets[k])
    for k in surfaces.keys():
      self.add_surface(k, surfaces[k])
  
  def __getitem__(self,s):
    from array import array
    from copy import deepcopy
    if type(s) in [int, long]:
      labs = [s]
    if type(s) is slice:
      if s.start != None: 
        start = s.start
      else: 
        start = 1
      stop  = s.stop
      if s.step != None: 
        step = s.step
      else: 
        step = 1
      labs = range(start,stop,step)
    if type(s) in [tuple,list]:  
      labs = []
      for a in s:
       if type(a) in [int, long]:labs.append(a)
       if type(a) is str:labs += self.sets[a.lower()].tolist()
    if type(s) is str:
      s = s.lower()
      if s in self.sets.keys():
        labs = self.sets[s]
      else:
        labs = []
    elabels = self.labels
    dtf = self.dtf
    dti = self.dti
    esets = self.sets
    connectivity = self.connectivity
    space = self.space
    name = self.name
    nlabels = self.nodes.labels
    nsets = self.nodes.sets
    N = Nodes(dti=dti, dtf=dtf)
    E = Mesh(nodes = N)
    for l in labs:
      toset = []
      i = elabels.index(l)
      for sk in esets.keys():
        if l in esets[sk]:
          toset.append(sk)
      E.add_element(label = l, connectivity = connectivity[i], space = space[i], name = name[i], toset = toset)
    nlabs = []
    for i in xrange(len(E.connectivity)):
      nlabs += E.connectivity[i].tolist()
    nlabs = [i for i in set(nlabs)]
    E.nodes = self.nodes[nlabs]
         
    return E      
  
  def __repr__(self):
    pattern = '<Mesh class instance: {} elements>'
    return pattern.format(len(self.labels))  
  def __str__(self):
    labels = self.labels
    c = self.connectivity
    s = self.space
    n = self.name
    out ='Mesh class instance:\nElements:\nLabel\tConnectivity\t\tSpace\tName\n'
    for i in xrange(len(labels)):
      out +=  '{0}\t{1}\t{2}\t{3}\n'.format(labels[i],c[i].tolist(),'{0}D'.format(s[i]),n[i])
    out+= 'Sets:\nLabel\tElements\n'
    for k in self.sets.keys():
      out +=  '{0}\t'.format(k)
      for n in self.sets[k]:
        out += '{0},'.format(n)
      out = out[0:-1] +'\n'
    
    return out
  def add_element(self,connectivity, space, label=None, name=None, toset=None):
    '''
    Adds an element.
    
    :param connectivity: element connectivity using node labels.
    :type connectivity: list of int > 0
    :param space: element embedded space, can be 1 for lineic element, 2 for surfacic element and 3 for volumic element.
    :type space: int in [1,2,3]
    :param name: element name used in fem code. 
    :type name: string
    :param toset: set(s) to which element is to be added. If a set does not exist, it is created. 
    :type toset: string or list of strings
    
    >>> from abapy.mesh import Mesh, Nodes
    >>> mesh = Mesh()
    >>> nodes = mesh.nodes
    >>> # Adding some nodes
    ... nodes.add_node(label = 1, x = 0. ,y = 0. , z = 0.)
    >>> nodes.add_node(label = 2, x = 1. ,y = 0. , z = 0.)
    >>> nodes.add_node(label = 3, x = 1. ,y = 1. , z = 0.)
    >>> nodes.add_node(label = 4, x = 0. ,y = 1. , z = 0.)
    >>> nodes.add_node(label = 5, x = 2. ,y = 0. , z = 0.)
    >>> nodes.add_node(label = 6, x = 2. ,y = 1. , z = 0.)
    >>> # Adding some elements
    ... mesh.add_element(label=1, connectivity = (1,2,3,4), space =2, name = 'QUAD4', toset='mySet' )
    >>> mesh.add_element(label=2, connectivity = (2,5,6,3), space =2, name = 'QUAD4', toset = ['mySet','myOtherSet'] )
    >>> print mesh
    Mesh class instance:
    Elements:
    Label	Connectivity		Space	Name
    1	[1L, 2L, 3L, 4L]	2D	QUAD4
    2	[2L, 5L, 6L, 3L]	2D	QUAD4
    Sets:
    Label	Elements
    myotherset	2
    myset	1,2'''
    
    from array import array
    from copy import copy
    dti, dtf = self.dti, self.dtf
    # Input verifications
    connect = array(dti,connectivity)
    if label == None: 
      if len(self.labels) != 0:
        label = int(max(self.labels)+1)
      else:
        label = 1
    try:
      label = int(label)
    except:
      raise Exception, 'element labels must be int.'
    if space not in [1,2,3]: raise Exception, 'space must be 1,2 or 3'
    if name == None: name = ''
    if type(name) is not str: raise Exception, 'name type must be str'  
    # inputs processing
    if label in self.labels: 
      print 'Info: element {0} already exists, nothing changed.'.format(label)
      return
    else:
      if len(self.labels) == 0:
        self.labels.append(label)
        self.connectivity.append(connect)
        self.space.append(space)
        self.name.append(name)
      else: 
        if label > max(self.labels):
          self.labels.append(label)
          self.connectivity.append(connect)
          self.space.append(space)
          self.name.append(name)
        else:
          dummy = copy(self.labels)
          dummy.append(label)
          i = sorted(dummy).index(label)
          del dummy # Is it necessary ?
          self.labels.insert(i,label)
          self.connectivity.insert(i,connect)
          self.space.insert(i,space)
          self.name.insert(i,name)
    if toset != None:
      if type(toset) is str:
        self.add_set(toset,[label])
      else:
        if '__getitem__' in dir(toset):
          for s in toset:
            self.add_set(s,[label])
  
  def drop_element(self,label):
    '''
    Removes one element to Mesh instance. The element is also removed from sets and surfaces, if a set or surface happens to be empty, it is also removed.
    
    :param label: element to be removed's label.
    :type label: int > 0
    
    >>> from abapy.indentation import ParamInfiniteMesh
    >>> from copy import copy
    >>> 
    >>> # Let's create a mesh containing a surface:
    ... m = ParamInfiniteMesh(Na = 2, Nb = 2)
    >>> print m.surfaces
    {'samp_surf': [('top_elem', 3)]}
    >>> elem_to_remove = copy(m.sets['top_elem'])
    >>> # Let's remove all elements in the surface:
    ... for e in elem_to_remove:
    ...   m.drop_element(e)
    ... # We can see that sets and surfaces are removed when they become empty
    ... 
    >>> print m.surfaces
    {}

    '''
    
    labels = self.labels
    conn = self.connectivity
    sets = self.sets
    surfaces = self.surfaces
    i = labels.index(label)
    labels.remove(label)
    conn.pop(i)
    for k in sets.keys():
      if label in sets[k]: sets[k].remove(label) 
      if len(sets[k]) == 0: 
        del sets[k]    
        for s in surfaces.keys():
          surf = surfaces[s]
          for i in xrange(len(surf)):
            if surf[i][0] == k:
              del surf[i] 
    for s in surfaces.keys():
      if surfaces[s] == []: del surfaces[s]          
    
  def drop_node(self, label):
    '''
    Drops a node from mesh.nodes instance. This method differs from to the nodes.drop_node element because it removes the node but also all elements containing the node in the mesh instance.
    
    :param label: node to be removed's label.
    :type label: int > 0
    
    .. plot:: example_code/mesh/Mesh-drop_node.py
     :include-source:
    '''
    self.nodes.drop_node(label)
    labels = self.labels
    conn = self.connectivity
    elem2remove = []
    for i in xrange(len(labels)):
      if label in conn[i]: elem2remove.append(labels[i])
    for l in elem2remove: self.drop_element(l)
        
      
    
          
  def add_set(self,label,elements):
    '''
    Adds a new set or appends elements to an existing set.
    
    :param label: set label to be added.
    :type label: string
    :param elements: element(s) that belong to the step.
    :type elements: int > 0 or list of int > 0
    
    >>> from abapy.mesh import Mesh, Nodes
    >>> mesh = Mesh()
    >>> nodes = mesh.nodes
    >>> # Adding some nodes
    >>> nodes.add_node(label = 1, x = 0. ,y = 0. , z = 0.)
    >>> nodes.add_node(label = 2, x = 1. ,y = 0. , z = 0.)
    >>> nodes.add_node(label = 3, x = 1. ,y = 1. , z = 0.)
    >>> nodes.add_node(label = 4, x = 0. ,y = 1. , z = 0.)
    >>> nodes.add_node(label = 5, x = 2. ,y = 0. , z = 0.)
    >>> nodes.add_node(label = 6, x = 2. ,y = 1. , z = 0.)
    >>> # Adding some elements
    >>> mesh.add_element(label=1, connectivity = (1,2,3,4), space =2, name = 'QUAD4')
    >>> mesh.add_element(label=2, connectivity = (2,5,6,3), space =2, name = 'QUAD4')
    >>> # Adding sets
    >>> mesh.add_set(label = 'niceSet', elements = 1)
    >>> mesh.add_set(label = 'veryNiceSet', elements = [1,2])
    >>> mesh.add_set(label = 'simplyTheBestSet', elements = 1)
    >>> mesh.add_set(label = 'simplyTheBestSet', elements = 2)
    >>> print mesh
    Mesh class instance:
    Elements:
    Label	Connectivity		Space	Name
    1	[1L, 2L, 3L, 4L]	2D	QUAD4
    2	[2L, 5L, 6L, 3L]	2D	QUAD4
    Sets:
    Label	Elements
    niceset	1
    veryniceset	1,2
    simplythebestset	1,2

    '''
    from array import array
    if type(label) is not str: raise Exception, 'set names must be strings.'
    label2 = label.lower()
    if label2 not in self.sets: self.sets[label2] = array(self.dti,[])
    if type(elements) == int or type(elements) == long: elements = [elements]
    if len(elements) == 0: return
    for i in elements:
      if i in self.labels: 
        self.sets[label2].append(i)
      else:
        print 'Info: element {0} does not exist, it was not added to set {1}.'.format(i,label2)
    self.sets[label2] = list(set(self.sets[label2]))
  
  def add_surface(self, label, description):
    '''
    Adds or expands an element surface (*i. e.* a group a element faces). Surfaces are used to define contact interactions in simulations.
    
    :param label: surface label.
    :type label: string
    :param description: list of ( element set label , face number ) tuples.
    :type description: list containing tuples each containing a string and an int
    
    >>> from abapy.mesh import RegularQuadMesh
    >>> mesh = RegularQuadMesh()
    >>> mesh.add_surface('topsurface', [ ('top', 1) ])
    >>> mesh.add_surface('topsurface', [ ('top', 2) ])
    >>> mesh.surfaces
    {'topsurface': [('top', 1), ('top', 2)]}
    '''
    
    if type(label) is not str: raise Exception, 'Surface labels must be string, got {0} instead'.format(type(label))
    if type(description) is not list: raise Exception, 'Surface description must be list, got {0} instead'.format(type(description))
    for i in xrange(len(description)):
      t = description[i]
      if len(t) != 2: raise Exception , 'Surface description elements must have 2 elements, got {0} instead'.format(len(t))
      if type(t[0]) is not str: raise Exception, '(In surface) Element sets keys must must be str, got {0} instead'.format(type(t[0]))
      if type(t[1]) not in [int, long]: raise Exception, 'Face numbers must keys must be int (generally in [1...6]), got {0} instead'.format(type(t[1]))
    if label not in self.surfaces.keys(): self.surfaces[label] = []
    surf = self.surfaces[label]
    for i in xrange(len(description)):
      t = description[i]
      tt = ( t[0] , t[1] )
      if tt not in surf: surf.append(tt)
        
  
  def add_field(self, field, label):
    """
    Add a field to the mesh.
    """
    self.fields[label] = field
    
    
  def dump2inp(self):
    '''
    Dumps the whole mesh (*i. e.* elements + nodes) to Abaqus INP format.
    
    :rtype: string
    
    >>> from abapy.mesh import Mesh, Nodes
    >>> mesh = Mesh()
    >>> nodes = mesh.nodes
    >>> # Adding some nodes
    >>> nodes.add_node(label = 1, x = 0. ,y = 0. , z = 0.)
    >>> nodes.add_node(label = 2, x = 1. ,y = 0. , z = 0.)
    >>> nodes.add_node(label = 3, x = 1. ,y = 1. , z = 0.)
    >>> nodes.add_node(label = 4, x = 0. ,y = 1. , z = 0.)
    >>> nodes.add_node(label = 5, x = 2. ,y = 0. , z = 0.)
    >>> nodes.add_node(label = 6, x = 2. ,y = 1. , z = 0.)
    >>> # Adding some elements
    >>> mesh.add_element(label=1, connectivity = (1,2,3,4), space =2, name = 'QUAD4')
    >>> mesh.add_element(label=2, connectivity = (2,5,6,3), space =2, name = 'QUAD4')
    >>> # Adding sets
    >>> mesh.add_set(label = 'veryNiceSet', elements = [1,2])
    >>> # Adding surfaces
    >>> mesh.add_surface(label = 'superNiceSurface', description = [ ('veryNiceSet', 2) ])
    >>> out = mesh.dump2inp()'''
    
    out = self.nodes.dump2inp()
    out += '**----------------------------------\n** ELEMENTS\n**----------------------------------\n'
    elTypes = set(self.name)
    elements = {}
    for elType in elTypes: elements[elType] = []
    for i in xrange(len(self.labels)): elements[self.name[i]].append(self.labels[i])
    for elType in elements.keys():
      els = elements[elType]
      if len(els) != 0:
        out +='*ELEMENT, TYPE={0}, ELSET={0}_ELEMENTS\n'.format(elType)
        pattern = '{0}, '
        for k in xrange(len(els)):
          label = els[k]
          i = self.labels.index(label)
          conn = self.connectivity[i]
          out += '  {0}, '.format(label) 
          for summit in conn:
            out += pattern.format(summit)
          out += '\n' 
    # ELEMENT SETS
    out +='**----------------------------------\n** ELEMENT SETS\n**----------------------------------\n'
    pattern = '*ELSET, ELSET={0}\n '
    pattern1 = ' {0},'
    sets = self.sets 
    for k in sets.keys():
      eset = sets[k]
      out += pattern.format(k.upper())
      nCount = 0
      for e in eset:
        out += pattern1.format(e)
        nCount += 1
        if nCount >= 9: 
          out += '\n'
          nCount = 0
      out += '\n'      
    # SURFACES
    out +='**----------------------------------\n** ELEMENT SURFACES\n**----------------------------------\n'
    pattern = '*SURFACE, TYPE=ELEMENT, NAME={0}\n'
    pattern1 = ' {0}, S{1}\n'
    surfaces = self.surfaces
    for k in surfaces.keys():
      surf = surfaces[k]
      out += pattern.format(k.upper())
      for surfset in surf:
        out += pattern1.format(surfset[0], surfset[1])
    return out[0:-1]
    
  def convert2tri3(self,mapping=None):
    '''
    Converts 2D elements to 3 noded triangles only. 
    
    :param mapping: gives the mapping of element name changes to be applied when elements are splitted. Example: mapping = {'CAX4':'CAX3'}
    :type mapping: dict with string keys and values  
    :rtype: Mesh instance containing only triangular elements.
      
    .. note:: This function was mainly developped to allow easy ploting in matplotlib using ``matplotlib.plyplot.triplot``, ``matplotlib.plyplot.tricontour`` and ``matplotlib.plyplot.contourf`` which rely on full triangle meshes. On a practical point of view, it easily used wrapped inside the ``abapy.Mesh.dump2triplot`` methods which rewrites connectivity in an easier to plot way.
      
    .. plot:: example_code/mesh/Mesh-convert2tri3.py
     :include-source:
    '''
    from array import array
    from copy import deepcopy
    dti = self.dti
    dtf = self.dtf
    nodes = deepcopy(self.nodes)
    #surfaces = deepcopy(self.surfaces)
    el2 = Mesh(nodes)
    for i in xrange(len(self.labels)):
      l = int(self.labels[i])
      c = self.connectivity[i]
      s = self.space[i]
      n = self.name[i]
      if s == 2:
        sets = []
        for sk in self.sets.keys():
          if l in self.sets[sk]: sets.append(sk)
        if mapping !=None:
          if n in mapping.keys(): n2 = mapping[n]
        else: 
          n2 = None
        if len(c) == 3:
          el2.add_element(connectivity = c,space = s, name = n, sets = sets)
        if len(c) == 4:
          el2.add_element(connectivity = [c[0],c[1],c[2]], space = s, name = n2, toset = sets)
          el2.add_element(connectivity = [c[2],c[3],c[0]], space = s, name = n2, toset = sets)
        if len(c) == 6:
          el2.add_element(connectivity = [c[0],c[3],c[5]], space = s, name = n2, toset = sets)
          el2.add_element(connectivity = [c[3],c[1],c[4]], space = s, name = n2, toset = sets)
          el2.add_element(connectivity = [c[3],c[4],c[5]], space = s, name = n2, toset = sets)
          el2.add_element(connectivity = [c[5],c[4],c[2]], space = s, name = n2, toset = sets)
        if len(c) == 8:
          el2.add_element(connectivity = [c[0],c[4],c[7]], space = s, name = n2, toset = sets)
          el2.add_element(connectivity = [c[4],c[1],c[5]], space = s, name = n2, toset = sets)
          el2.add_element(connectivity = [c[5],c[2],c[6]], space = s, name = n2, toset = sets)
          el2.add_element(connectivity = [c[7],c[6],c[3]], space = s, name = n2, toset = sets)
          el2.add_element(connectivity = [c[7],c[4],c[6]], space = s, name = n2, toset = sets)
          el2.add_element(connectivity = [c[4],c[5],c[6]], space = s, name = n2, toset = sets)
    return el2
  def get_border(self, xmin = None, xmax = None, ymin = None, ymax = None, zmin = None, zmax = None):
    import numpy as np
    '''
    Returns the list of edges belonging to the border of the meshed domain. Edges are given as x and y lists with None separator for faster ploting in ``matplotlib.pyplot``. 
    
    :rtype: 3 lists of coordinates directly plotable in matplotlib
    
    '''
    x,y,z = self.nodes.x, self.nodes.y, self.nodes.z
    conn = self.connectivity
    borderEdges = []
    xe,ye, ze =[],[],[]
    bi = borderEdges.index
    ni = self.nodes.labels.index
    for ne in xrange(len(self.labels)):
      if self.space[ne] == 2:
        el = conn[ne]
        el = [n for n in el]+[el[0]]
        for i in xrange(len(el)-1):
          edge = (el[i],el[i+1])
          revert = (edge[1],edge[0]) 
          if revert not in borderEdges and edge not in borderEdges:
            borderEdges.append(edge)
          else:
            if revert in borderEdges: 
              borderEdges.pop(bi(revert))
            else: 
              borderEdges.pop(bi(edge))
      if self.space[ne] == 3:
        el = conn[ne]
        if len(el) == 4:
          edges = [ (el[0],el[1]) , (el[1],el[2]), (el[2],el[0]), (el[0],el[3]), (el[1],el[3]) , (el[2],el[3]) ]
        if len(el) == 6:
          edges = [ (el[0],el[1]), (el[1],el[2]), (el[2],el[0]), (el[3],el[4]), (el[4],el[5]), (el[5],el[3]), (el[0],el[3]), (el[1],el[4]), (el[2],el[5])]
        if len(el) == 8:
          edges = [ (el[0],el[1]), (el[1],el[2]), (el[2],el[3]), (el[3],el[0]), (el[4],el[5]), (el[5],el[6]), (el[6],el[7]), (el[7],el[4]), (el[0],el[4]), (el[1],el[5]), (el[2],el[6]), (el[3],el[7])]
        for edge in edges:
          revert = (edge[1],edge[0]) 
          b0 = edge not in borderEdges
          b1 = revert not in borderEdges
          if b0 and b1:
            borderEdges.append(edge)
          else:
            if b0 == False: 
              borderEdges.pop(bi(edge))
            if b1 == False: 
              borderEdges.pop(bi(revert))
    if xmin == None: xmin = min(x)
    if xmax == None: xmax = max(x)
    if ymin == None: ymin = min(y)
    if ymax == None: ymax = max(y)
    if zmin == None: zmin = min(z)
    if zmax == None: zmax = max(z)
    
    def test_func(x,y,z):
      '''
      Tests if the node is in the bb.
      '''
      return (x >= xmin) and  (x <= xmax) and  (y >= ymin) and  (y <= ymax) and  (z >= zmin) and  (z <= zmax)
        
    for edge in borderEdges:
      n0, n1 = ni(edge[0]), ni(edge[1])
      x0, y0, z0 = x[n0], y[n0], z[n0]
      x1, y1, z1 = x[n1], y[n1], z[n1] 
      if (test_func(x0,y0,z0) or test_func(x0,y0,z0)):
        xe += [x0,x1,np.nan]
        ye += [y0,y1,np.nan]
        ze += [z0,z1,np.nan]
    xe = np.ma.array( xe, mask = np.isnan(xe) )
    ye = np.ma.array( ye, mask = np.isnan(ye) )
    ze = np.ma.array( ze, mask = np.isnan(ze) )
    return xe, ye, ze
  
  def get_edges(self, xmin = None, xmax = None, ymin = None, ymax = None, zmin = None, zmax = None):
    '''
    Returns the list of edges composing the meshed domain. Edges are given as x and y lists with None separator for faster ploting in ``matplotlib.pyplot``. 
    
    :rtype: 3 lists of coordinates directly plotable in matplotlib
    '''
    import numpy as np
    x,y,z = self.nodes.x, self.nodes.y, self.nodes.z
    conn = self.connectivity
    borderEdges = []
    xe,ye, ze =[],[],[]
    bi = borderEdges.index
    ni = self.nodes.labels.index
    for ne in xrange(len(self.labels)):
      if self.space[ne] == 2:
        el = conn[ne]
        el = [n for n in el]+[el[0]]
        for i in xrange(len(el)-1):
          edge = (el[i],el[i+1])
          revert = (edge[1],edge[0]) 
          if revert not in borderEdges and edge not in borderEdges:
            borderEdges.append(edge)
      if self.space[ne] == 3:
        el = conn[ne]
        if len(el) == 4:
          edges = [ (el[0],el[1]) , (el[1],el[2]), (el[2],el[0]), (el[0],el[3]), (el[1],el[3]) , (el[2],el[3]) ]
        if len(el) == 6:
          edges = [ (el[0],el[1]), (el[1],el[2]), (el[2],el[0]), (el[3],el[4]), (el[4],el[5]), (el[5],el[3]), (el[0],el[3]), (el[1],el[4]), (el[2],el[5])]
        if len(el) == 8:
          edges = [ (el[0],el[1]), (el[1],el[2]), (el[2],el[3]), (el[3],el[0]), (el[4],el[5]), (el[5],el[6]), (el[6],el[7]), (el[7],el[4]), (el[0],el[4]), (el[1],el[5]), (el[2],el[6]), (el[3],el[7])]
        for edge in edges:
          revert = (edge[1],edge[0]) 
          if revert not in borderEdges and edge not in borderEdges:
            borderEdges.append(edge)
    if xmin == None: xmin = min(x)
    if xmax == None: xmax = max(x)
    if ymin == None: ymin = min(y)
    if ymax == None: ymax = max(y)
    if zmin == None: zmin = min(z)
    if zmax == None: zmax = max(z)
    
    def test_func(x,y,z):
      '''
      Tests if the node is in the bb.
      '''
      return (x >= xmin) and  (x <= xmax) and  (y >= ymin) and  (y <= ymax) and  (z >= zmin) and  (z <= zmax)
        
    for edge in borderEdges:
      n0, n1 = ni(edge[0]), ni(edge[1])
      x0, y0, z0 = x[n0], y[n0], z[n0]
      x1, y1, z1 = x[n1], y[n1], z[n1] 
      if (test_func(x0,y0,z0) or test_func(x0,y0,z0)):
        xe += [x0,x1,np.nan]
        ye += [y0,y1,np.nan]
        ze += [z0,z1,np.nan]
    xe = np.ma.array( xe, mask = np.isnan(xe) )
    ye = np.ma.array( ye, mask = np.isnan(ye) )
    ze = np.ma.array( ze, mask = np.isnan(ze) )
    return xe, ye, ze
   
  def drop_set(self,label):
    '''Goal: drops a set without removing elements and nodes.
    Inputs:
      * label: set label to be dropped, must be string.   
    '''
    if label in self.sets.keys():
      del self.sets[label]
    else:
      print 'Info: sets {0} does not exist and cannot be dropped.'.format(label)     
   
  def extrude(self, N = 1 , l = 1., quad = False, mapping = {}):
    '''
    Extrudes a mesh in z direction. The method is made to be applied to 2D mesh, it may work on shell elements but may lead to inside out elements. 
    
    :param N: number of ELEMENTS along z, must > 0.
    :type N: int
    :param l: length of the extrusion along z, should be > 0 to avoid inside out elements.
    :type l: float
    :param quad: specifies if quadratic elements should be used instead of linear elements (default). Doesn't work yet. Linear and quadratic elements should not be mixed in the same mesh.
    :type quad: boolean
    :param mapping: gives the way to translate element names during extrusion. Example: {'CAX4':'C3D8','CAX3':'C3D6'}. If 2D element name is not in the mapping, names will be chosen in the basic continuum elements used by Abaqus: 'C3D6' and 'C3D8'.   
    :type mapping: boolean
    
    .. plot:: example_code/mesh/Mesh-extrude.py
      :include-source:    
    '''
    
    from copy import deepcopy
    import numpy as np
    from array import array
    delta = float(l)/N # Delta to be applied along z axis
    inodes = self.nodes # Input nodes
    omesh = Mesh(nodes = deepcopy(inodes)) # Output Mesh
    tempmesh = Mesh(nodes = deepcopy(inodes)) # Temporary Mesh
    # Selection of 2D elements
    for i in xrange(len(self.labels)):
      label = self.labels[i]
      conn = self.connectivity[i]
      space = self.space[i]
      name = self.name[i]
      #print label,conn,space, name
      if space == 2:
        if quad == False:
          if len(conn) in [3,4]: 
            tempmesh.add_element(label = label, connectivity = conn,space = space, name = name)
    for sk in self.sets.keys():
      tempmesh.add_set(sk,self.sets[sk])
    #nNodes, nElements = len(tempmesh.nodes.labels), len(tempmesh.labels) 
    nmax, emax = max(self.nodes.labels), max(self.labels)
    # Creating new nodes and elements
    if quad == False:
      for i in xrange(N):
        # Nodes
        tn =  tempmesh.nodes
        te =  tempmesh
        tns = tempmesh.nodes.sets
        tes = tempmesh.sets
        for nn in xrange(len(tn.labels)):
          oldLabel = tn.labels[nn]
          label = oldLabel + (i+1) * nmax
          x = tn.x[nn]
          y = tn.y[nn]
          z = tn.z[nn] + delta * (i+1)
          omesh.nodes.add_node(label,x,y,z)
         
        # Elements
        for ne in xrange(len(te.labels)):
          oldLabel = te.labels[ne]
          label = oldLabel + i * emax
          conn  = te.connectivity[ne]
          conn = [c + (i) * nmax  for c in conn] + [c + (i+1) * nmax  for c in conn]
          name  = te.name[ne]
          if name in mapping.keys():
            name = mapping[name]
          else: 
            name = 'C3D{0}'.format(len(conn))
          omesh.add_element(label = label,connectivity = conn,space = 3,name = name)
          for sk in tes.keys():
            s = tes[sk]
            if oldLabel in s:
              omesh.add_set(sk,label)
    # Managing sets
    if quad == False:
      ins = self.nodes.sets
      onodes = omesh.nodes
      ies = self.sets
      omesh = omesh
      for k in ins.keys(): # Nodes
        s = ins[k]
        os = []
        for i in xrange(1, N+1):
          os += [ l + i * nmax for l in s]
        onodes.add_set(k, deepcopy(os)) 
      for k in ies.keys(): # Elements
        s = ies[k]
        os = []
        for i in xrange(N):
          os += [ l + i * emax for l in s]
        omesh.add_set(k, deepcopy(os))   
    # Managing surfaces:
    set_pattern = '{0}_f{1}_{2}'
    if quad == False:
      isurf = self.surfaces
      for k in isurf.keys():
        s = isurf[k]
        elems, faces, etype = [], [], []
        for couple in s:
          labels = self.sets[couple[0]]
          face = couple[1]
          for l in labels:
            elems.append(l)
            faces.append(face)
            i = self.labels.index(l)
            etype.append(len(self.connectivity[i]))   
        F = {3:[],4:[],5:[],6:[] }
        for i in xrange(len(elems)):
          l = elems[i]
          f = faces[i]
          t = etype[i]
          F[f+2] += [l + emax * layer for layer in xrange(N)]
        surf = []  
        for i in xrange(3,7):
          j = 0
          set_label = set_pattern.format(k,i,j)
          while set_label in self.sets.keys():
            j += 1
            set_label = set_pattern.format(k,i,j)
          if F[i] != []: 
            omesh.add_set(set_label, F[i])
            surf.append((set_label, i))
        omesh.add_surface(k, surf)  
    return omesh
    
  def sweep(self,N = 1,sweep_angle = 45., quad = False , mapping={}, extrude = False ):
    '''
    Sweeps a mesh in around z axis. The method is made to be applied to 2D mesh, it may work on shell elements but may lead to inside out elements. 
    
    :param N: number of ELEMENTS along z, must > 0.
    :type N: int
    :param sweep_angle: sweep angle around the z axis in degrees. Should be > 0 to avoid inside out elements.
    :type sweep_angle: float
    :param quad: specifies if quadratic elements should be used instead of linear elements (default). Doesn't work yet. Linear and quadratic elements should not be mixed in the same mesh.
    :type quad: boolean
    :param mapping: gives the way to translate element names during extrusion. Example: {'CAX4':'C3D8','CAX3':'C3D6'}. If 2D element name is not in the mapping, names will be chosen in the basic continuum elements used by Abaqus: 'C3D6' and 'C3D8'.   
    :type mapping: dictionary
    :param extrude: if True, this param will modify the transformation used to produce the sweep. The result will be a mixed form of sweep and extrusion useful to produce pyramids. When using this option, the sweep angle must be lower than 90 degrees.
    :type extrude: boolean 
    
    .. plot:: example_code/mesh/Mesh-sweep.py
      :include-source:    
    '''
    
    from copy import deepcopy
    import numpy as np
    from array import array
    from math import sin, cos, tan, pi, radians
    delta = radians(float(sweep_angle)/N) # Delta to be applied along z axis
    inodes = self.nodes # Input nodes
    omesh = Mesh(nodes = deepcopy(inodes)) # Output Mesh
    
    tempmesh = Mesh(nodes = deepcopy(inodes)) # Temporary Mesh
    # Selection of 2D elements
    for i in xrange(len(self.labels)):
      label = self.labels[i]
      conn = self.connectivity[i]
      space = self.space[i]
      name = self.name[i]
      #print label,conn,space, name
      if space == 2:
        if quad == False:
          if len(conn) in [3,4]: 
            tempmesh.add_element(label = label, connectivity = conn,space = space, name = name)
    for sk in self.sets.keys():
      tempmesh.add_set(sk,self.sets[sk])
    nmax, emax = max(self.nodes.labels), max(self.labels)
    # Creating new nodes and elements
    def transform_sweep(x0, y0, z0, layer, delta):
      x = x0 * cos(delta*(layer + 1 )) + z0 * sin(delta * (layer + 1))
      y = y0
      z = z0 * cos(delta*(layer + 1 )) - x0 * sin(delta * (layer + 1))
      return x, y, z
    def transform_extrude(x0, y0, z0, layer, delta):
      x = x0 
      y = y0
      z = z0 - x0 * tan(delta * (layer + 1))
      return x, y, z
    if extrude == True:
      transform = transform_extrude
    else:
      transform = transform_sweep
    if quad == False:
      for layer in xrange(N):
        # Nodes
        tn =  tempmesh.nodes
        te =  tempmesh
        tns = tempmesh.nodes.sets
        tes = tempmesh.sets
        for nn in xrange(len(tn.labels)):
          oldLabel = tn.labels[nn]
          label = oldLabel + (layer+1) * nmax
          x0, y0, z0 = tn.x[nn], tn.y[nn], tn.z[nn]
          if (x0 == 0. and z0 == 0.) == False: 
            x, y, z = transform(x0, y0, z0, layer, delta)
            omesh.nodes.add_node(label,x,y,z)
            
            
        # Elements
        def get_nodesOnAxis(conn):
          nodesOnAxis = [False for l in oldconn]
          for i in xrange(len(oldconn)):
            indice = tn.labels.index(oldconn[i])
            x,y,z = tn.x[indice], tn.y[indice], tn.z[indice]
            if x == 0. and z == 0.: nodesOnAxis[i] = True
          return nodesOnAxis
          
        for ne in xrange(len(te.labels)):
          oldLabel = te.labels[ne]
          label = oldLabel + layer * emax
          oldconn, newconn  = te.connectivity[ne], None
          nodesOnAxis = get_nodesOnAxis(oldconn)
          nnoa = nodesOnAxis.count(True)
          if nnoa == 0:
            newconn = [c + (layer+1) * nmax  for c in oldconn] + [c + layer * nmax  for c in oldconn]
          if nnoa == 2:  
            if len(oldconn) == 3:
              """
              newconn = []
              for i in xrange(3):
              
                if nodesOnAxis[i] == True: 
                  out_of_axis = i
                  newconn.append(oldconn[i])
                else:
                  newconn.append(oldconn[i] + layer * nmax)
              newconn.append(oldconn[out_of_axis] + (layer+1) * nmax)
              """
              while nodesOnAxis != [True, False, True]:
                oldconn = [oldconn[2],oldconn[0],oldconn[1]]
                nodesOnAxis = get_nodesOnAxis(oldconn)
              newconn = [oldconn[0], oldconn[1] + layer * nmax, oldconn[2], oldconn[1] + (layer +1) * nmax]
              
            if len(oldconn) == 4:
              while nodesOnAxis != [True, False, False, True]:
                oldconn = [oldconn[-1],oldconn[0],oldconn[1],oldconn[2]]
                nodesOnAxis = get_nodesOnAxis(oldconn)
              newconn = [oldconn[0], layer * nmax  + oldconn[1], (layer+1) * nmax  + oldconn[1], oldconn[3], layer * nmax  + oldconn[2], (layer+1) * nmax  + oldconn[2]] 
          if newconn != None: 
            name  = te.name[ne]
            if name in mapping.keys():
              name = mapping[name]
            else: 
              name = 'C3D{0}'.format(len(newconn))
            omesh.add_element(label = label,connectivity = newconn,space = 3,name = name)
          else:
            print 'element {0} has not been sweeped'.format(oldLabel)
    # Managing sets
    if quad == False:
      ins = self.nodes.sets
      onodes = omesh.nodes
      ies = self.sets
      omesh = omesh
      for k in ins.keys(): # Nodes
        s = ins[k]
        os = []
        for i in xrange(1, N+1):
          for l in s:
            l2 = l + i * nmax
            if l2 in omesh.nodes.labels: os.append(l2)
        onodes.add_set(k, deepcopy(os)) 
      for k in ies.keys(): # Elements
        s = ies[k]
        os = []
        for i in xrange(N):
          for l in s:
            l2 = l + i * emax
            if l2 in omesh.labels: os.append(l2)
        omesh.add_set(k, deepcopy(os))   
      # front and back node sets
      frontnodes = []
      for i in xrange(len(inodes.labels)):
        oldLabel = inodes.labels[i]
        x, y, z = inodes.x[i], inodes.y[i], inodes.z[i] 
        if x != 0. or z != 0.: frontnodes.append(oldLabel)
      omesh.nodes.add_set('front_nodes',frontnodes)
      omesh.nodes.add_set('back_nodes',[nl + nmax * N for nl in frontnodes])
    # Managing surfaces:
    set_pattern = '{0}_f{1}_{2}'
    if quad == False:
      isurf = self.surfaces
      for k in isurf.keys():
        s = isurf[k]
        elems, faces, etype = [], [], []
        for couple in s:
          labels = self.sets[couple[0]]
          face = couple[1]
          for l in labels:
            elems.append(l)
            faces.append(face)
            i = self.labels.index(l)
            etype.append(len(self.connectivity[i]))   
        F = {1:[],2:[],3:[],4:[],5:[],6:[] }
        for i in xrange(len(elems)):
          l = elems[i]
          f = faces[i]
          t = etype[i]
          pos = omesh.labels.index(l)
          t2 = len(omesh.connectivity[pos])
          if t == 3:
            if t2 == 6: F[f+2] += [l + emax * layer for layer in xrange(N)]
            if t2 == 4: 
              #pos0 = pos = self.labels.index(l)
              #oldconn = self.connectivity[pos0]
              #newconn = omesh.connectivity[pos]
              #nnoa = newconn[-1]-nmax
              #innoa = oldconn.index(nnoa)
              if f == 1 : F[2] += [l + emax * layer for layer in xrange(N)]
              if f == 2 : F[3] += [l + emax * layer for layer in xrange(N)]
              #if f == 3 and innoa in [3,1] : F[4] += [l + emax * layer for layer in xrange(N)]
              
          if t == 4:
            if t2 == 8: F[f+2] += [l + emax * layer for layer in xrange(N)]
            if t2 == 6: 
              if f == 1 : F[1] += [l + emax * layer for layer in xrange(N)]
              if f == 2 : F[4] += [l + emax * layer for layer in xrange(N)]
              if f == 3 : F[2] += [l + emax * layer for layer in xrange(N)]
              
        surf = []  
        for i in xrange(1,7):
          j = 0
          set_label = set_pattern.format(k,i,j)
          while set_label in self.sets.keys():
            j += 1
            set_label = set_pattern.format(k,i,j)
          if F[i] != []: 
            omesh.add_set(set_label, F[i])
            surf.append((set_label, i))
        omesh.add_surface(k, surf)      
    if sweep_angle == 360. : omesh.simplify_nodes()
    return omesh
      
    
  def dump2vtk(self, path = None):
    '''
    Dumps the mesh to the VTK format. VTK format can be visualized using Mayavi2 or Paraview. This method is particularly useful for 3D mesh. For 2D mesh, it may be more efficient to work with matplotlib using methods like: get_edges, get_border and dump2triplot.
    
    :param path: if None, return a string containing the VTK data. If not, must be a path to a file where the data will be written.
    :rtype: string or None.
    
    .. plot:: example_code/mesh/Mesh-dump2vtk.py
      :include-source:
      
    * VTK output: :download:`Mesh-dump2vtk.vtk <example_code/mesh/Mesh-dump2vtk.vtk>` 
    * Paraview plot: 
    
    .. image:: example_code/mesh/Mesh-dump2vtk.png 
    
    '''
    out = '# vtk DataFile Version 2.0\nUnstructured Grid Example\nASCII\nDATASET UNSTRUCTURED_GRID\n'
    # Nodes
    nodes = self.nodes
    nnodes = len(nodes.labels)
    ni = nodes.labels.index
    out += 'POINTS {0} float\n'.format(nnodes)
    for i in xrange(nnodes):
      out += '{0} {1} {2}\n'.format(nodes.x[i], nodes.y[i], nodes.z[i])
    # Elements:
    
    esize = 0
    nel = len(self.labels)
    for i in xrange(nel): esize += 1 + len(self.connectivity[i])
    out += 'CELLS {0} {1}\n'.format(nel, esize)
    eTypes = 'CELL_TYPES {0}\n'.format(nel)
    pattern = '{0} '
    pattern2 = '{0}\n'
    for i in xrange(nel):
      c = self.connectivity[i]
      s = self.space[i]
      lc = len(c)
      out += pattern.format(lc)
      for n in c: 
        out += pattern.format(ni(n))
      out += '\n'
      if s == 2:
        if lc == 3: ecode = 5
        if lc == 4: ecode = 9
      if s == 3:
        if lc == 4: ecode = 10
        if lc == 6: ecode = 13
        if lc == 8: ecode = 12
      eTypes += pattern2.format(ecode)
    out += eTypes
    #Fields
    fields = self.fields
    nfields = {} # node fields
    efields = {} # element fields
    for key in fields.keys():
      if fields[key].position == "node": nfields[key] = fields[key]
      if fields[key].position == "element": efields[key] = fields[key]
    header = True
    if len(nfields.keys()) != 0:
      for key in nfields.keys():      
        out += fields[key].dump2vtk(name = key, header = header)
        header = False
    header = True
    if len(efields.keys()) != 0:
      for key in nfields.keys():      
        out += fields[key].dump2vtk(name = key, header = header)
        header = False   
    if path == None:
      return out
    else:
      f = open(path, "wb")
      f.write(out)
      f.close()
      
      
  def dump2triplot(self, use_3D = False):
    '''
    Allows any 2D mesh to be triangulized and formated in a suitable way to be used by triplot, tricontour and tricontourf in matplotlib.pyplot. This is the best way to produce clean 2D plots of 2D meshs. Returns 4 arrays/lists: x, y and z coordinates of nodes and triangles connectivity. It can be directly used in matplotlib.pyplot using:
    
    :rtype: 4 lists
    
    >>> import matplotlib.pyplot as plt
    >>> from abapy.mesh import RegularQuadMesh
    >>> plt.figure()
    >>> plt.axis('off')
    >>> plt.gca().set_aspect('equal')
    >>> mesh = RegularQuadMesh(N1 = 10 , N2 = 10)
    >>> x,y,z,tri = mesh.dump2triplot()
    >>> plt.triplot(x,y,tri)
    >>> plt.show()'''
    from array import array
    import numpy as np
    from copy import copy
    if use_3d == False:
      dti = self.dti
      m3 = self.convert2tri3()
      c0 = m3.connectivity
      ni = m3.nodes.labels.index
      c = []
      for t0 in c0:
        t = array(dti,[])
        for n in t0:
          t.append(ni(n))
        c.append(copy(t))
      return np.array(m3.nodes.x), np.array(m3.nodes.y), np.array(m3.nodes.x), np.array(c)
    else:
      n = self.nodes
      conn = np.array(self.connectivity)
      space = np.array(self.space)
      labels = n.labels
      conn2D = conn[np.where(space == 2)[0]]
      conn3D = conn[np.where(space == 3)[0]]
      if use_3D == False:
        nodes = np.array([n.x, n.y]).transpose()
        verts = [[nodes[ labels.index(i) ] for i in c ] for c in conn2D]
      else: 
        def order_conn(c):
          c = np.array(c)
          m = c.min()
          p = np.where(c == m)[0][0]
          n = len(c)
          o = np.arange(n)
          c = c[o-n+p]
          if c[-1] < c[1]: c = c[-o]
          return c.tolist()
        nodes = np.array([n.x, n.y, n.z]).transpose()
        faces = []
        for c in conn3D:
          if len(c) == 8: # HEXAHEDRON
            local_faces = [
            [0,1,2],
            [3,2,1],
            [0,4,1],
            [1,5,4],
            [4,7,6],
            [6,5,4],
            [7,3,2],
            [2,6,7],
            [1,2,6],
            [6,5,1],
            [0,3,7],
            [7,4,0]]
            
          for lf in local_faces:
            fc = order_conn([c[i] for i in lf])
            fc = [labels.index(p) for p in fc]
            if fc in faces:
              loc = faces.index(fc)
              faces.pop(loc)
            else:
              faces.append(fc)
          
      return np.array(self.nodes.x), np.array(self.nodes.y), np.array(self.nodes.z), np.array(faces) 
        
  
  def replace_node(self, old, new):
    '''
    Replaces a node of given label (old) by another existing node (new). This version of ``replace_node`` differs from the version of the ``Nodes`` class because it also updates elements connectivity. When working with mesh (an not only nodes), this version should be used.
    
    :param old: node label to be replaced.
    :type old: int > 0
    :param new: node label of the node replacing old.
    :type new: int > 0
    
    >>> from abapy.mesh import RegularQuadMesh
    >>> N1, N2 = 1,1
    >>> mesh = RegularQuadMesh(N1, N2)
    >>> mesh.replace_node(1,2)
    Info: element 1 maybe have become degenerate due du node replacing.
    >>> print mesh 
    Mesh class instance:
    Elements:
    Label	Connectivity		Space	Name
    1	[2L, 4L, 3L]	2D	QUAD4
    Sets:
    Label	Elements
    '''
    
    elabels = self.labels
    conn = self.connectivity
    esets = self.sets
    nlabels = self.nodes.labels
    # Some verifications
    if old not in nlabels or new not in nlabels: raise Exception, 'Both node labels must exist.'
    self.nodes.replace_node(old, new)
    # Let's proceed
    for i in xrange(len(elabels)):
      c = conn[i]
      if old in c:
        pos = c.index(old)
        c.pop(pos)
        if new not in c:
          c.insert(pos,new)
        else:
          el = elabels[i]
          print 'Info: element {0} maybe have become degenerate due du node replacing.'.format(el)  

  '''
  def simplify_nodes(self, crit_distance = 1.e-10):
    from copy import copy
    """
    Looks for duplicated nodes and removes them.
    
    :param crit_istance: critical under which two nodes are considered as identical.
    :type crit_distance: float > 0.                                                            
    """
    nodes = self.nodes
    nlabels = nodes.labels
    nlabels2 = copy(nlabels) # nlabels may change in the process
    for nlabel in nlabels2:
      if nlabel in nlabels:
        closest_label, distance = nodes.closest_node(nlabel)
        if distance <= crit_distance:
          self.replace_node(closest_label, nlabel)
  '''
  
  def simplify_nodes(self, crit_distance = 1.e-10):
    from copy import copy
    import numpy as np
    '''
    Looks for duplicated nodes, removes them and updates connectivity.
    
    :param crit_istance: critical under which two nodes are considered as identical.
    :type crit_distance: float > 0.                                                            
    
    '''
    
    nodes = self.nodes
    x, y, z = np.array(nodes.x), np.array(nodes.y), np.array(nodes.z)
    points = np.array([x,y,z]).transpose()
    neighbors = get_neighbors(points, crit_dist = crit_distance)
    nlabels = np.array(nodes.labels)
    for i in xrange(len(neighbors)):
      new_label = nlabels[i]
      n = neighbors[i]
      if n != []:
        old_labels = nlabels[n]
        for old_label in old_labels:
          self.replace_node(old_label, new_label)
          
  def union(self,other_mesh, crit_distance = None, simplify = True):
    '''
    Computes the union of 2 Mesh instances. The second operand's labels are increased to be compatible with the first. All sets are kepts and merged if they share the same name. Nodes which are too close (< crit_distance) are merged. If crit_distance is None, the defautl value value of ``simplify_mesh`` is used.
    
    :param other_mesh: mesh to be added to current mesh.
    :type other_mesh: ``Mesh`` instance
    :param crit_distance: critical distance under which nodes are considered identical.
    :type crit_distance: float > 0
    '''
    from array import array as n_array
    if isinstance(other_mesh, Mesh) == False: raise Exception, 'other_mesh should be Mesh instance, got {0}'.format(type(other_mesh))
    dti, odti = self.dti, other_mesh.dti
    dtf, odtf = self.dtf, other_mesh.dtf
    if dti == 'I' or odti == 'I': 
      new_dti = 'I'
    else:
      new_dti = 'H'
    if dtf == 'd' or odtf == 'd': 
      new_dtf = 'd'
    else:
      new_dtf = 'f'
    nodes = self.nodes
    if len(nodes.labels) != 0:
      max_nlabel = max(nodes.labels)
    else:
      max_nlabel = 0
    if len(self.labels)!= 0:  
      max_elabel = max(self.labels)
    else:
      max_elabel = 0
    onodes = other_mesh.nodes
    if len(onodes.labels)!= 0:
      for i in xrange(len(onodes.labels)):
        l = onodes.labels[i]
        node = onodes[l]
        toset = node.sets.keys()
        x, y, z = node.x[0], node.y[0], node.z[0]
        nodes.add_node(label = l + max_nlabel, x = x, y = y, z = z, toset = toset  )
      for i in xrange(len(other_mesh.labels)):
        l = other_mesh.labels[i]
        element = other_mesh[l]
        toset = element.sets.keys()
        conn = [n + max_nlabel for n in element.connectivity[0] ]
        space = element.space[0]
        name = element.name[0]
        self.add_element(label = l + max_elabel, connectivity = conn, space = space, name = name, toset = toset  )
      if simplify:
        if crit_distance == None:
          self.simplify_nodes()    
        else:
          self.simplify_nodes(crit_distance = crit_distance)
  
  def apply_reflection(self,point = (0., 0., 0.), normal = (1., 0., 0.)):
    '''
    Applies a reflection symmetry to the mesh instance. The reflection plane is defined by a point and a normal direction.
    
    :param point: coordinates of a point of the reflection plane.
    :type point: tuple or list containing 3 floats
    :param normal: normal vector to the reflection plane
    :type normal: tuple or list containing 3 floats
    :rtype: ``Mesh`` instance
    
    ..note: This method can lead to coherence problems with surfaces, this problem will be addressed in the future. Surfaces are removed by this operation untill this problem is solved.
    
    .. plot:: example_code/mesh/Mesh_apply_reflection.py
     :include-source:
    '''  
    from copy import copy
    out = copy(self)
    out.nodes = out.nodes.apply_reflection(point=point, normal=normal)
    out.surfaces={}
    for c in out.connectivity: c.reverse()
    return out
       
  def centroids(self):
    """
    Returns a dictionnary containing the coordinates of all the nodes belonging to earch element.
    
    .. plot:: example_code/mesh/Mesh-centroids.py
     :include-source:
    """
    import numpy as np
    nodes = self.nodes
    conn = self.connectivity
    space = self.space
    
    def tri_area(vertices):
      u = vertices[0]
      v = vertices[1]
      w = vertices[2]
      return np.linalg.norm(np.cross( v-u, w-u)) / 2.
    
    def tetra_area(vertices):
      u = vertices[0]
      v = vertices[1]
      w = vertices[2]
      x = vertices[3]
      return abs(np.cross(v-u, w-u).dot(x-u)) / 6. 
    
    def simplex_centroid(vertices):
      return vertices.sum(axis = 0) / len(vertices)
    
    


    centroids = np.zeros([len(conn), 3])
    for n in xrange(len(conn)):
      c = conn[n]
      s = space[n]
      vertices = np.zeros([len(c), 3])
      for i in xrange(len(c)):
        loc = nodes.labels.index(c[i])
        vertices[i,0] = nodes.x[loc]
        vertices[i,1] = nodes.y[loc]
        vertices[i,2] = nodes.z[loc]
        
      if s == 2: # 2D
        if len(c) == 3:  # Triangle
          centroids[n] = simplex_centroid(vertices)
        if len(c) == 4:  # Quadrangle
          t0 =  vertices[[0,1,2]]
          t1 =  vertices[[2,3,0]] 
          a0 = tri_area(t0)  
          a1 = tri_area(t1)
          c0 = simplex_centroid(t0)
          c1 = simplex_centroid(t1)
          centroids[n] = (c0 * a0 + c1 * a1) / (a0 + a1)   
      if s == 3: # 3D
        if len(c) == 4: # Tretrahedron
          centroids[n] = simplex_centroid(vertices)
        if len(c) == 6: #Prism
          t0 =  vertices[[0,1,2,3]]
          t1 =  vertices[[1,2,3,4]]
          t2 =  vertices[[2,3,4,5]] 
          a0 = tetra_area(t0)  
          a1 = tetra_area(t1)
          a2 = tetra_area(t2)
          c0 = simplex_centroid(t0)
          c1 = simplex_centroid(t1)
          c2 = simplex_centroid(t2)
          centroids[n] = (c0 * a0 + c1 * a1 + c2 * a2) / (a0 + a1 + a2)
        if len(c) == 8: #Hexahedron
          t0 =  vertices[[0,1,3,4]]
          t1 =  vertices[[1,2,3,4]]
          t2 =  vertices[[2,3,7,4]]
          t3 =  vertices[[2,6,7,4]]
          t4 =  vertices[[1,5,2,4]]
          t5 =  vertices[[2,5,6,4]]
           
          a0 = tetra_area(t0)  
          a1 = tetra_area(t1)
          a2 = tetra_area(t2)
          a3 = tetra_area(t3)  
          a4 = tetra_area(t4)
          a5 = tetra_area(t5)
          c0 = simplex_centroid(t0)
          c1 = simplex_centroid(t1)
          c2 = simplex_centroid(t2)
          c3 = simplex_centroid(t3)
          c4 = simplex_centroid(t4)
          c5 = simplex_centroid(t5)
          centroids[n] = (c0 * a0 + c1 * a1 + c2 * a2 + c3 * a3 + c4 * a4 + c5 * a5) / (a0 + a1 + a2 + a3 + a4 + a5)       
    return centroids
    
  def volume(self):
    """
    Returns a dictionnary containing the volume of all the elements.
    
    .. plot:: example_code/mesh/Mesh-volume.py
     :include-source:
    """
    import numpy as np
    nodes = self.nodes
    conn = self.connectivity
    space = self.space
    
    def tri_area(vertices):
      u = vertices[0]
      v = vertices[1]
      w = vertices[2]
      return np.linalg.norm(np.cross( v-u, w-u)) / 2.
    
    def tetra_area(vertices):
      u = vertices[0]
      v = vertices[1]
      w = vertices[2]
      x = vertices[3]
      return abs(np.cross(v-u, w-u).dot(x-u)) / 6. 
     


    volume = np.zeros([len(conn)])
    for n in xrange(len(conn)):
      c = conn[n]
      s = space[n]
      vertices = np.zeros([len(c), 3])
      for i in xrange(len(c)):
        loc = nodes.labels.index(c[i])
        vertices[i,0] = nodes.x[loc]
        vertices[i,1] = nodes.y[loc]
        vertices[i,2] = nodes.z[loc]
        
      if s == 2: # 2D
        if len(c) == 3:  # Triangle
          volume[n] = tri_area(vertices)
        if len(c) == 4:  # Quadrangle
          t0 =  vertices[[0,1,2]]
          t1 =  vertices[[2,3,0]] 
          a0 = tri_area(t0)  
          a1 = tri_area(t1)
          volume[n] = a0 + a1   
      if s == 3: # 3D
        if len(c) == 4: # Tretrahedron
          volume[n] = tetra_area(vertices)
        if len(c) == 6: #Prism
          t0 =  vertices[[0,1,2,3]]
          t1 =  vertices[[1,2,3,4]]
          t2 =  vertices[[2,3,4,5]] 
          a0 = tetra_area(t0)  
          a1 = tetra_area(t1)
          a2 = tetra_area(t2)
          volume[n] = a0 + a1 + a2
        if len(c) == 8: #Hexahedron
          t0 =  vertices[[0,1,3,4]]
          t1 =  vertices[[1,2,3,4]]
          t2 =  vertices[[2,3,7,4]]
          t3 =  vertices[[2,6,7,4]]
          t4 =  vertices[[1,5,2,4]]
          t5 =  vertices[[2,5,6,4]]
          a0 = tetra_area(t0)  
          a1 = tetra_area(t1)
          a2 = tetra_area(t2)
          a3 = tetra_area(t3)  
          a4 = tetra_area(t4)
          a5 = tetra_area(t5)
          volume[n] = a0 + a1 + a2 + a3 + a4 + a5   
    return volume  
    
    
  def faces(self, use_3D = False):
    """
    Returns the vertices that can be used to plot the mesh.
    """
    import numpy as np
    n = self.nodes
    conn = np.array(self.connectivity)
    space = np.array(self.space)
    labels = n.labels
    conn2D = conn[np.where(space == 2)[0]]
    conn3D = conn[np.where(space == 3)[0]]
    if use_3D == False:
      nodes = np.array([n.x, n.y]).transpose()
      verts = [[nodes[ labels.index(i) ] for i in c ] for c in conn2D]
    else: 
      def order_conn(c):
        c = np.array(c)
        m = c.min()
        p = np.where(c == m)[0][0]
        n = len(c)
        o = np.arange(n)
        c = c[o-n+p]
        if c[-1] < c[1]: c = c[-o]
        return c.tolist()
      nodes = np.array([n.x, n.y, n.z]).transpose()
      faces = []
      for c in conn3D:
        if len(c) == 8: # HEXAHEDRON
          local_faces = [
          [0,1,2,3],
          [0,4,5,1],
          [4,7,6,5],
          [7,3,2,6],
          [1,5,6,2],
          [0,4,7,3],]
        if len(c) == 6: # PRISM
          local_faces = [
          [0,1,2],
          [3,4,5],
          [0,1,4,3],
          [1,2,5,4],
          [0,2,5,3],]  
        for lf in local_faces:
          fc = order_conn([c[i] for i in lf])
          fc = [labels.index(p) for p in fc]
          if fc in faces:
            loc = faces.index(fc)
            faces.pop(loc)
          else:
            faces.append(fc)
      verts = [[nodes[i ] for i in c ] for c in faces]      
    return verts
    
  def dump2polygons(self, edge_color = "black", edge_width = 1., face_color = None, use_3D = False):
    """
    Returns 2D elements as matplotlib poly collection.
    
    :param edge_color: edge color.
    :param edge_width: edge width.
    :param face_color: face color.
    :param use_3D: True for 3D polygon export.
    
    .. plot:: example_code/mesh/Mesh-dump2polygons.py
     :include-source:
    
    .. plot:: example_code/mesh/Mesh-dump2polygons_3D.py
     :include-source:
    
    """  
    from matplotlib import cm
    import numpy as np
    import matplotlib.collections as collections
    verts = self.faces(use_3D = use_3D)
    if use_3D == False:
      if face_color == None:
        patches = collections.LineCollection(verts, 
                            color = edge_color, 
                            linewidth = edge_width) 
                            
      else:
        patches = collections.PolyCollection(verts, 
                            edgecolor = edge_color, 
                            linewidth = edge_width, 
                            facecolor = face_color)                      
    else:
      import mpl_toolkits.mplot3d as a3 
      if face_color == None:
        patches = a3.art3d.Line3DCollection(verts,
                           edgecolor = edge_color, 
                           linewidth = edge_width)   
      else:
        patches = a3.art3d.Poly3DCollection(verts,
                           edgecolor = edge_color, 
                           linewidth = edge_width, 
                           facecolor = face_color)                        
    return patches
  
  def draw(self, ax, field_func = None, disp_func = None, cmap = None, cmap_levels = 20, cbar_label = 'Field', cbar_orientation = 'horizontal', edge_color = "black", edge_width = 1., node_style = "k.", node_size = 1., contour = False, contour_colors = "black", alpha = 1.):
    """
    Draws a 2D mesh in a given matplotlib axes instance.
    
    :param ax: matplotlib axes instance.
    :param field_func: a function that defines how to used existing fields to produce a FieldOutput instance.
    :type field_func: function or None
    :param disp_func: a function that defines how to used existing fields to produce a VectorFieldOutput instance used as a diplacement field.
    :type disp_func: function
    :param cmap: matplotlib colormap.
    :param cmap_levels: number of levels in the colormap
    :param cbar_label: colorbar label.
    :type cbar_label: string
    :param cbar_orientation: "horizontal" or "vertical".
    :param edge_color: valid matplotlib color for the edges of the mesh.
    :param edge_width: mesh edge width.
    :param node_style: nodes plot style.
    :param node_size: nodes size.
    :param contour: plot field contour.
    :type contour: boolean
    :param contour_colors: contour colors to use, colormap of fixed color.
    :param alpha: alpha lvl of the gradiant plot.
    
    .. plot:: example_code/mesh/Mesh-draw.py
     :include-source:
    
    """
    from matplotlib import pyplot as plt
    mesh = copy.copy(self)
    if disp_func != None: 
      U = disp_func(mesh.fields)
      mesh.nodes.apply_displacement(U)
    patches = mesh.dump2polygons()
    bb = mesh.nodes.boundingBox()
    patches.set_linewidth(edge_width)
    patches.set_edgecolor(edge_color)
    ax.set_aspect("equal")
    if field_func != None:
      if cmap == None:
        from matplotlib import cm
        cmap = cm.jet
      X, Y, Z, tri = mesh.dump2triplot()
      field = field_func(mesh.fields)
      grad = ax.tricontourf(X, Y, tri, field.data, cmap_levels, cmap = cmap, alpha = alpha)
      bar = plt.colorbar(grad, orientation = cbar_orientation)
      bar.set_label(cbar_label)
      if contour:
        ax.tricontour(X, Y, tri, field.data, cmap_levels, colors = contour_colors)   
    ax.add_collection(patches)
    ax.plot(mesh.nodes.x, mesh.nodes.y, node_style, markersize = 1.)
  
  
  def node_set_to_surface(self, surface, nodeSet):
    """
    Builds a surface from a node set.
    
    :param surface: surface label
    :type surface: string
    :param nodeSet: nodeSet label
    :type nodeSet: string
    
    .. plot:: example_code/mesh/Mesh-node_set_to_surface.py
     :include-source:
    """
    nodes = self.nodes
    nset = set(nodes.sets[nodeSet])
    f1, f2, f3, f4 = [], [], [], [] 
    for i in xrange(len(self.labels)):
      c = list(self.connectivity[i])
      inter = nset & set(c)
      ind = set([c.index(j) for j in inter])
      if self.space[i] == 2:
        if len(inter) >= 2:
          if set([0,1]) <= ind: f1.append(self.labels[i])
          if set([1,2]) <= ind: f2.append(self.labels[i])
          if set([2,3]) <= ind: f3.append(self.labels[i])
          if set([3,0]) <= ind: f4.append(self.labels[i])
    if len(f1) != 0: 
      self.add_set(surface+"_f1", f1)
      self.add_surface(surface, [(surface+"_f1", 1)])
    if len(f2) != 0: 
      self.add_set(surface+"_f2", f2)
      self.add_surface(surface, [(surface+"_f2", 2)])
    if len(f3) != 0: 
      self.add_set(surface+"_f3", f3)
      self.add_surface(surface, [(surface+"_f3", 3)])
    if len(f4) != 0: 
      self.add_set(surface+"_f4", f4)
      self.add_surface(surface, [(surface+"_f4", 4)])        
       
  
def RegularQuadMesh(N1=1, N2=1, l1=1.,l2=1.,name='QUAD4',dtf='f',dti='I'):
  '''Generates a 2D regular quadrangle mesh.
  
  :param N1: number of elements respectively along y.
  :type N1: int > 0
  :param N2: number of elements respectively along y.
  :type N2: int > 0
  :param l1: length of the mesh respectively along x.
  :type l1: float
  :param l2: length of the mesh respectively along y.
  :type l2: float
  :param name: elements names, for example 'CPS4'.
  :type name: string
  :param dti: int data type in array.array
  :type dti: 'I', 'H'
  :param dtf: float data type in array.array
  :type dtf: 'f', 'd'
  :rtype: Mesh instance  

  
  .. plot:: example_code/mesh/RegularQuadMesh.py
     :include-source:
  '''
  import numpy as np
  from array import array as array
  x,y = np.meshgrid(np.linspace(0,l1,N1+1),np.linspace(0,l2,N2+1))
  X = np.reshape(x,(N1+1)*(N2+1))
  Y = np.reshape(y,(N1+1)*(N2+1))
  Cx,Cy = np.meshgrid(np.arange(1,N1+1),(N1+1)*np.arange(0,N2))
  C = np.reshape(Cx,N1*N2)+np.reshape(Cy,N1*N2) 
  mesh = Mesh(nodes = Nodes(dtf=dtf,dti=dti))
  nodes = mesh.nodes
  for i in xrange(len(X)): nodes.add_node(None,X[i],Y[i],0.)
  for i in xrange(len(C)): 
    mesh.add_element(connectivity = (C[i],C[i]+1,C[i]+N1+2,C[i]+N1+1), space = 2, name = name)
  nodes.add_set('bottomleft', 1)
  nodes.add_set('bottomright', N1+1)
  nodes.add_set('topleft', (N1+1)*N2+1)
  nodes.add_set('topright',(N1+1)*(N2+1))
  nodes.add_set('bottom', range(1,N1+2))
  nodes.add_set('top', range((N1+1)*N2+1, (N1+1)*(N2+1)+1))
  nodes.add_set('left', range(1,(N1+1)*N2+2,N1+1))
  nodes.add_set('right', range(N1+1,(N1+1)*(N2+1)+1,N1+1))
  return mesh

def UnitTransition(name='CAX4', l1 = 1., l2 =1.):
  m = Mesh()
  n = m.nodes
  n.add_node(label = 1, x = 0., y=0.)
  n.add_node(label = 2, x = 0.5, y=0.)
  n.add_node(label = 3, x = 1., y=0.)
  n.add_node(label = 4, x = 0., y=0.5)
  n.add_node(label = 5, x = 0.25, y=0.5)
  n.add_node(label = 6, x = 0.75, y=0.5)
  n.add_node(label = 7, x = 1., y=0.5)  
  n.add_node(label = 8, x = 0., y=1.)
  n.add_node(label = 9, x = 0.25, y=1.)
  n.add_node(label = 10, x = 0.5, y=1.)
  n.add_node(label = 11, x = 0.75, y=1.) 
  n.add_node(label = 12, x = 1., y=1.)
  m.add_element(connectivity = [1,2,5,4], label = 1, name = name, space = 2)
  m.add_element(connectivity = [2,10,9,5], label = 2, name = name, space = 2)
  m.add_element(connectivity = [2,6,11,10], label = 3, name = name, space = 2)
  m.add_element(connectivity = [2,3,7,6], label = 4, name = name, space = 2)
  m.add_element(connectivity = [4,5,9,8], label = 5, name = name, space = 2)  
  m.add_element(connectivity = [6,7,12,11], label = 6, name = name, space = 2) 
  def function(x, y, z, labels):
    ux = x * (l1 - 1.)
    uy = y * (l2 - 1.)
    uz = 0. * z 
    return ux, uy, uz
  u = m.nodes.eval_vectorFunction(function)
  m.nodes.apply_displacement(u)
  return m

def TransitionMesh(N1 = 4, N2 = 2, l1 = 1., l2 = 1., direction = "y+", name = 'CAX4', crit_distance = 1.e-6 ):
  """
  A mesh transition to manage them all...
  
  :param N1: starting number of elements, must be multiple of 4.
  :type N1: int
  :param N2: ending number of elements, must be lower than N1 and multiple of 2.
  :type N2: int
  :param l1: length of the mesh in the x direction.
  :type l1: float
  :param l2: length of the mesh in the y direction.
  :type l2: float
  :param direction: direction of mesh. Must be in ("x+", "x-", "y+", "y-").
  :type direction: str
  :param name: name of the element in the export procedures.
  :type name: str
  :param crit_distance: critical distance in union process.
  :type crit_distance: float
  
  .. plot:: example_code/mesh/TransitionMesh.py
     :include-source:
  """
  tx, ty = 0., 0.
  N1u, N2u = 0, N1
  k = 1.
  lx, ly = 2., 1. 
  m = Mesh()
  while N2u > N2:
    while N1u < N1:
      m1 = UnitTransition(l1 = k*lx, l2 = k*ly, name = name)
      m1.nodes.translate(x = tx*k*lx, y = ty)
      m.union(m1, crit_distance = crit_distance)
      N1u += 4
      tx  += 1.
    k *= 2.  
    N1 /= 2
    N1u = 0  
    tx = 0.
    ty -= k*ly
    N2u /= 2
  def function(x, y, z, labels):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max() 
    ux = -x + (x-xmin)/(xmax-xmin)
    uy = -y + (y-ymin)/(ymax-ymin)
    uz = 0. * z 
    return ux, uy, uz
  u = m.nodes.eval_vectorFunction(function)
  m.nodes.apply_displacement(u)  
  if direction != "y+":
    if direction == "y-":
      def function2(x, y, z, labels):
        ux = 1. - 2 * x
        uy = 1. - 2 * y
        uz = 0. * z
        return ux, uy, uz
    if direction == "x+":
      def function2(x, y, z, labels):
        ux = -x + y
        uy = -y - x + 1.
        uz = 0. * z
        return ux, uy, uz
    if direction == "x-":
      def function2(x, y, z, labels):
        ux = -x - y
        uy = -y + x -1.
        uz = 0. * z
        return ux, uy, uz    
    u2 = m.nodes.eval_vectorFunction(function2)
    u2 = m.nodes.eval_vectorFunction(function2)
    m.nodes.apply_displacement(u2)      
  def function(x, y, z, labels):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max() 
    ux = -x + (x-xmin)/(xmax-xmin)*l1 
    uy = -y + (y-ymin)/(ymax-ymin)*l2
    uz = 0. * z 
    return ux, uy, uz
  u = m.nodes.eval_vectorFunction(function)
  m.nodes.apply_displacement(u)  
  return m  
 
def RegularQuadMesh_like(x_list = [0., 1.], y_list = [0., 1.],name='QUAD4',dtf='f',dti='I'):
  '''Generates a 2D regular quadrangle mesh from 2 lists of positions. This version of RegularQuadMesh is an alternative to the normal one in some cases where fine tuning of x, y positions is required.
  
  :param x_list: list of x values
  :type x_list: list, array.array or numpy.array
  :param y_list: list of y values
  :type y_list: list, array.array or numpy.array
  :param name: elements names, for example 'CPS4'.
  :type name: string
  :param dti: int data type in array.array
  :type dti: 'I', 'H'
  :param dtf: float data type in array.array
  :type dtf: 'f', 'd'
  :rtype: Mesh instance  

  
  '''
  import numpy as np
  from array import array as array
  x,y = np.meshgrid(x_list,y_list)
  N1 = len(x_list)-1
  N2 = len(y_list)-1
  X = np.reshape(x,(N1+1)*(N2+1))
  Y = np.reshape(y,(N1+1)*(N2+1))
  Cx,Cy = np.meshgrid(np.arange(1,N1+1),(N1+1)*np.arange(0,N2))
  C = np.reshape(Cx,N1*N2)+np.reshape(Cy,N1*N2) 
  mesh = Mesh(nodes = Nodes(dtf=dtf,dti=dti))
  nodes = mesh.nodes
  for i in xrange(len(X)): nodes.add_node(None,X[i],Y[i],0.)
  for i in xrange(len(C)): 
    mesh.add_element(connectivity = (C[i],C[i]+1,C[i]+N1+2,C[i]+N1+1), space = 2, name = name)
  nodes.add_set('bottomleft', 1)
  nodes.add_set('bottomright', N1+1)
  nodes.add_set('topleft', (N1+1)*N2+1)
  nodes.add_set('topright',(N1+1)*(N2+1))
  nodes.add_set('bottom', range(1,N1+2))
  nodes.add_set('top', range((N1+1)*N2+1, (N1+1)*(N2+1)+1))
  nodes.add_set('left', range(1,(N1+1)*N2+2,N1+1))
  nodes.add_set('right', range(N1+1,(N1+1)*(N2+1)+1,N1+1))
  return mesh


def get_neighbors(points, crit_dist = 0.1, max_hits = 1):
  '''
  Internal use function performing nearest neighbor search using KDTree algorithm.
  '''
  import scipy.spatial as spatial
  t = spatial.cKDTree(data = points)
  neighbors, hits = [], []
  for i in xrange(len(t.data)):
    dist, loc = t.query(points[i], k=max_hits+1)
    dist = dist <= crit_dist  
    neighbors.append([])
    n = neighbors[-1]
    for k in xrange(len(loc)):
      l = loc[k]
      d = dist[k]
      if d and (l > i) and (l not in hits): 
        n.append(l)  
        hits.append(l)
  return neighbors 
  

