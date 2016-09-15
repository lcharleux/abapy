'''
Indentation
===========
'''



def ParamInfiniteMesh(Na = 10, Nb=10, l = 1., core_name = 'CAX4', add_shell = True, shell_name = 'CINAX4', dti = 'I', dtf = 'd'):

  '''
  Returns a mesh dedicated to 2D/Axisymmetrical indentation. It is composed of a core of quadrangle elements and a shell of infinite elements. The core is divided into three zones. The center is core_1, the right is core_2 and the bottom is core_3. Core_1 is a Na x Na square mesh whereas core_2 and core_3 are respectively Nb x Na and Na x Nb structured meshes that have been transformed to be connected and guaranty element size progression.
  
  :param Na: number of elements per side in core_1
  :type Na: int > 1
  :param Nb: number of radial elements in core_2 and core_3 
  :type Nb: int > 0
  :param l: core_1 size
  :type l: float > 0
  :param core_name: element name in core
  :type core_name: string
  :param add_shell: True if the shell of infinite elements is to be used, False if not.
  :type add_shell: boolean
  :param shell_name: element name in shell, should be infinite
  :type shell_name: string  
  '''
  from mesh import RegularQuadMesh, RegularQuadMesh_like
  from numpy import zeros_like, ones_like, linspace, where
  from copy import copy
  def disp2(x, y, z, labels):
    ux = zeros_like(x)
    uy = (x - l + ux) * y/l
    uz = zeros_like(x)
    return ux, uy, uz
  def disp3(x, y, z, labels):
    uy = zeros_like(x)
    ux = -(y + l + uy) * x/l
    uz = zeros_like(x)
    return ux, uy, uz  
  def conn_permutation(conn): # permutes nodes in connectivity to have a particular orientation. Flips CLOCKWISE.
    from copy import copy
    conn2 = copy(conn)
    for i in xrange(len(conn)): conn2[i] = conn[i-1]
    return conn2
  m1 = RegularQuadMesh_like(linspace(0.,l,Na+1),linspace(-l,0.,Na+1), name = core_name, dtf = dtf, dti = dti )
  axis = copy(m1.nodes.sets['left'])
  top = copy(m1.nodes.sets['top'])
  m1.nodes.sets = {}
  m1.add_set('core',m1.labels)
  m1.add_set('core_1',m1.labels)
  m1.add_set('top_elem',range( Na * (Na-1)+1, Na**2+1  ))
  m1.nodes.add_set('core',m1.nodes.labels)
  x2 = [l]
  for i in xrange(Nb): x2.append(x2[-1] * Na/(Na-1))
  y2 = linspace(-l,0,Na+1)
  m2 = RegularQuadMesh_like(x_list = x2, y_list = y2 , name = core_name, dtf = dtf, dti = dti )
  m2.nodes.sets = {}
  m2.add_set('core',m2.labels)
  m2.add_set('core_2',m2.labels)
  m2.add_set('top_elem',range( Nb * (Na-1)+1, Na*Nb+1  ))
  m2.nodes.add_set('core',m2.nodes.labels)
  y3 = [-l]
  for i in xrange(Nb): y3.append(y3[-1] * Na/(Na-1))
  y3.sort()
  x3 = linspace(0.,l,Na+1)
  m3 = RegularQuadMesh_like(x_list = x3, y_list = y3 , name = core_name, dtf = dtf, dti = dti )
  m3.nodes.sets = {}
  m3.add_set('core',m3.labels)
  m3.nodes.add_set('core',m3.nodes.labels)
  m3.add_set('core_3',m3.labels)
  if add_shell:
    m4 = RegularQuadMesh_like(x_list = [x2[-1], 2*x2[-1]], y_list = y2 , name = shell_name, dtf = dtf, dti = dti )
    m4.nodes.sets = {}
    m4.add_set('shell',m4.labels)
    m5 = RegularQuadMesh_like(x_list = x3, y_list = [2*y3[0], y3[0]] , name = shell_name, dtf = dtf, dti = dti )
    m5.nodes.sets = {}
    m5.add_set('shell',m5.labels)
    for i in xrange(len(m4.labels)): # permutates 1 time
      m4.connectivity[i] = conn_permutation(m4.connectivity[i])   
    for i in xrange(len(m5.labels)): # permutates 2 times
      m5.connectivity[i] = conn_permutation(m5.connectivity[i])
      m5.connectivity[i] = conn_permutation(m5.connectivity[i]) 
    m2.union(m4)
    m3.union(m5)
  U2 = m2.nodes.eval_vectorFunction(disp2)
  m2.nodes.apply_displacement(U2)
  U3 = m3.nodes.eval_vectorFunction(disp3)
  m3.nodes.apply_displacement(U3)
  m1.union(m2)
  m1.union(m3)
  m1.nodes.add_set_by_func('top', lambda x, y, z, labels: y == 0.)
  m1.nodes.add_set_by_func('axis', lambda x, y, z, labels: x == 0.)
  m1.add_set('all_elements', m1.labels)
  m1.add_surface('samp_surf',[ ('top_elem',3) ])
  return m1

def IndentationMesh(Na = 8, Nb = 8, Ns = 4, Nf = 2 , l =1., name = 'CAX4', dtf = 'f', dti = 'I'):
  '''
  An indentation oriented full quad mesh.
  
  :param Na: number of elements along x axis in the finely meshed contact zone. *Must be power of 2*.
  :type Na: int
  :param Nb: number of elements along y axis in the finely meshed contact zone. *Must be power of 2*.
  :type Nb: int
  :param Ns: number of radial elements in the shell. 
  :type Ns: int
  :param Nf: number of orthoradial elements in each half shell. Must be > 0.
  :type Nf: int
  :param l: length of the square zone. 
  :type l: float.
  :param name: name of the elements. Note that this mesh if full quad so only one name is required.
  :type name: string
  :param dtf: float data type in ``array.array``, 'd' for float64 or 'f' for float32.
  :type dtf: string
  :param dti: int data type in ``array.array``, 'I' for unsignedint32 or 'H' for unsignedint16 (dangerous in some cases).
  :type dti: string
  
  .. plot:: example_code/indentation/IndentationMesh.py
      :include-source:    
  '''
  from abapy.mesh import Mesh, RegularQuadMesh, RegularQuadMesh, RegularQuadMesh_like
  from math import radians
  from numpy import zeros_like, ones_like, linspace, sin, cos, pi, tan, array, logical_and
  from copy import copy

  def disp2(x, y, z, labels):
    xa = l
    ya = y
    xmax, xmin = max(x), min(x)
    ymin, ymax = min(y), max(y)
    theta = -y/ymin * pi/4.
    xb = xmax * cos(theta)
    yb = xmax * sin(theta)
    alpha = (x - xmin) / (xmax - xmin)
    x2 = alpha * xb + (1- alpha) * xa
    y2 = alpha * yb + (1- alpha) * ya
    return x2 -x, y2 - y,  zeros_like(x)
   
  def disp3(x, y, z, labels):
    ya = -l
    xa = x
    xmax, xmin = max(x), min(x)
    ymin, ymax = min(y), max(y)
    theta = x/xmax * pi/4.- pi/2.
    xb = -ymin * cos(theta)
    yb = -ymin * sin(theta)
    alpha = (-y + ymax) / (-ymin + ymax)
    x2 = alpha * xb + (1- alpha) * xa
    y2 = alpha * yb + (1- alpha) * ya
    return x2 -x, y2 - y,  zeros_like(x)

  def UnitTransition(name='CAX4'):
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
    return m

  def Transition(N = 4, l1 = 1., l2 = 1., name = 'CAX4', crit_distance = 1.e-6 ):
    from copy import deepcopy
    N = N / 4
    m = UnitTransition(name)
    m0 = deepcopy(m)
    if N > 1:
      for i in xrange(1,N):
        m2 = deepcopy(m0)
        m2.nodes.translate(x=i)
        m.union(m2, crit_distance = crit_distance)
    x = m.nodes.x
    y = m.nodes.y
    for i in xrange(len(x)):
      x[i] = x[i] / float(N) * l1
      y[i] = l2 * y[i]
    return m

  def PolarTransition(N = 4, radius = 1., theta0 = -90., theta1 = 0., size = None, name = 'CAX4' ):
    mesh = Transition(N = N, name = 'CAX4')
    t0, t1 = radians(theta0), radians(theta1)
    if size == None: size = (t1-t0 ) * radius /  N 
    def dispTrans(x, y, z, labels):
      ux = -x + (radius + 2 * size *  (1-y)) * cos(t0 + (t1-t0) * x)
      uy = -y + (radius + 2 * size *  (1-y)) * sin(t0 + (t1-t0) * x ) 
      uz = 0. * z
      return ux, uy, uz
    Utrans = mesh.nodes.eval_vectorFunction(dispTrans)
    mesh.nodes.apply_displacement(Utrans)
    return mesh
    
  
    
  def PolarShell(radius = 1., N=2 , Ns = 4, theta0 = -90., theta1 = 0., k = None, name = 'CAX4'):
    t0, t1 = radians(theta0), radians(theta1)
    t = (t1-t0)/N
    if k == None: k = t / (1.- t/2.)
    y1 = [radius]
    for i in xrange(Ns): y1.append(y1[-1] * (k+1))
    y1.reverse()
    x1 = linspace(0., 1., N+1)
    mesh = RegularQuadMesh_like(x_list = x1, y_list = y1 , name = name, dtf = dtf, dti = dti )
    mesh.sets= {}
    mesh.nodes.sets={}
    def dispTrans(x, y, z, labels):
      
      ux = -x + y * cos(t0 + (t1-t0) * x)
      uy = -y + y * sin(t0 + (t1-t0) * x ) 
      uz = 0. * z
      return ux, uy, uz
    U = mesh.nodes.eval_vectorFunction(dispTrans)
    mesh.nodes.apply_displacement(U)
    return mesh
  
  Nmin, Nmax = min(Na, Nb) ,max(Na, Nb) 
  crit_distance = l/Nmax/100.
  m1 = RegularQuadMesh_like(linspace(0.,l,Na+1),linspace(-l,0.,Nb+1), name = name, dtf = dtf, dti = dti )
  m1.sets = {}
  m1.nodes.sets={}
  m1.add_set('surface_elements',range( Na * (Nb-1)+1, Na*Nb+1  ))
  m1.add_set('core_elements',m1.labels)
  m1.nodes.add_set('core_nodes',m1.nodes.labels)
  x2 = [l]
  for i in xrange(Nmin): x2.append(x2[-1] * Nmin/(Nmin-1))
  y2 = linspace(-l,0,Nb+1)
  m2 = RegularQuadMesh_like(x_list = x2, y_list = y2 , name = name, dtf = dtf, dti = dti )
  y3 = [-l]
  for i in xrange(Nmin): y3.append(y3[-1] * Nmin/(Nmin-1))
  y3.sort()
  x3 = linspace(0.,l,Na+1)
  m3 = RegularQuadMesh_like(x_list = x3, y_list = y3 , name = name, dtf = dtf, dti = dti )
  U2 = m2.nodes.eval_vectorFunction(disp2)
  m2.nodes.apply_displacement(U2)
  U3 = m3.nodes.eval_vectorFunction(disp3)
  m3.nodes.apply_displacement(U3)
  m2.sets={}
  m2.nodes.sets={}
  m2.add_set('surface_elements',range( Nmin * (Nb-1)+1, Nmin*Nb+1  ))
  m2.add_set('core_elements',m2.labels)
  m2.nodes.add_set('core_nodes',m2.nodes.labels)
  m3.sets={}
  m3.nodes.sets={}
  m3.add_set('core_elements',m3.labels)
  m3.nodes.add_set('core_nodes',m3.nodes.labels)
  m1.union(m2, crit_distance = crit_distance)
  m1.union(m3, crit_distance = crit_distance)
  surface_radius = max(m1.nodes.x)
  Nta, Ntb = Na, Nb 
  while Nta > Nf and Ntb > Nf:
    radius = max(m1.nodes.x)
    size = (pi/4. ) * radius / min(Nta, Ntb)
    transa = PolarTransition( N = Nta, theta0=-90., theta1=-45., radius = radius, size = size)
    transb = PolarTransition( N = Ntb, theta0=-45., theta1=0., radius = radius, size = size)
    transa.union(transb, crit_distance = crit_distance)
    transa.add_set('shell_elements',transa.labels)
    transa.nodes.add_set('shell_nodes',transa.nodes.labels)
    m1.union(transa, crit_distance = crit_distance)
    Nta, Ntb = Nta /2, Ntb/2

  radius = max(m1.nodes.x)
  t = (pi/4.)/min(Nta, Ntb)
  k = t / (1.- t/2.)
  shella = PolarShell(radius = radius, N = Nta, Ns = Ns, theta0=-90., theta1=-45., k = k)
  shellb = PolarShell(radius = radius, N = Ntb, Ns = Ns, theta0=-45., theta1=  0.,  k = k)
  shella.union(shellb, crit_distance = crit_distance)
  shella.add_set('shell_elements',shella.labels)
  shella.nodes.add_set('shell_nodes',shella.nodes.labels)
  m1.union(shella, crit_distance = crit_distance)
  # Correcting some rounding errors:
  n1 = m1.nodes
  for i in xrange(len(n1.x)):
    if n1.x[i] < crit_distance: n1.x[i] = 0.
    if n1.y[i] > -crit_distance: n1.y[i] = 0.
  # Managing sets
  m1.nodes.add_set_by_func('top_nodes', lambda x, y, z, labels: logical_and(y == 0., x <= surface_radius))
  m1.nodes.add_set_by_func('surface_nodes', lambda x, y, z, labels: logical_and(y == 0., x <= surface_radius))
  m1.nodes.add_set_by_func('axis_nodes', lambda x, y, z, labels: x == 0.)
  m1.nodes.add_set('all_nodes',m1.nodes.labels)
  m1.add_surface('surface_faces',[ ('surface_elements',3) ])
  m1.add_set('all_elements', m1.labels)
  m1.nodes.add_set_by_func('tip_node',lambda x, y, z, labels: logical_and(y == 0., x == 0.))
  m1.nodes.add_set_by_func('ref_node',lambda x, y, z, labels: logical_and(y == min(m1.nodes.y), x==0.))
  radius = max(m1.nodes.x)
  error = 0.001
  m1.nodes.add_set_by_func('bottom_nodes', lambda x, y, z, labels: (x**2+y**2)**.5 > (1-error)*radius ) 
  
  return m1

class RigidCone2D:
  '''
  A rigid cone usable in 2D and Axisymmetric simulations.
  
  :param half_angle: half_angle in DEGREES.
  :type half_angle: float > 0.
  :param width: width of the indenter
  :type width: float > 0.
  :param summit_position: position of the summit in a 2D space.
  :type summit_position: tuple or list containing two floats.
  '''
  
  def __init__(self, half_angle= 70.3, width=10.,summit_position = (0., 0.)):
    self.set_half_angle(half_angle)
    self.set_width(width)
    self.set_summit_position(summit_position)
        
  def set_summit_position(self, summit_position):
    '''
    Sets the position of the indenter.
    
    :param summit_position: position of the summit in a 2D space.
    :type summit_position: tuple or list containing two floats.
    '''
    if len(summit_position) != 2: raise Exception, 'summit_position length must be 2, got {0} instead.'.format(len(summit_position))
    self.summit_position = (float(summit_position[0]) , float(summit_position[1]) )
  
  def set_half_angle(self, half_angle = 70.3):
    '''
    Sets the half angle of the indenter. Default is equivalent to modified Berkovich indenter in volume.
    
    :param half_angle: half_angle in DEGREES.
    :type half_angle: float > 0.
    '''
    if type(half_angle) not in [float, int, long]:
      raise Exception, 'half angle must be float, got {0}'.format(type(half_angle))
    self.half_angle = abs(float(half_angle))
  def set_width(self,width):
    '''
    Sets the width of the indenter.
    
    :param width: width
    :type width: float > 0.
    '''
    if type(width) not in [float, int, long]:
      raise Exception, 'width must be float, got {0}'.format(type(width))
    self.width = abs(float(width))
  
  
     
    
  def dump2inp(self):
    '''
    Dumps to Abaqus INP format.
    
    :rtype: string
    '''
    from math import radians, tan
    w = self.width
    ha = self.half_angle
    pattern = '''*NODE, NSET = REFERENCE
1,0.,1.,0.
*NSET, NSET=AXIS
1
*NSET, NSET=BOTTOM
1
*NSET, NSET=TIP
1
*SURFACE, TYPE = SEGMENTS, NAME = IND_SURF
START, {0}, {1}
LINE, 0., 0.
*RIGID BODY, REF NODE = REFERENCE, ANALYTICAL SURFACE = IND_SURF'''
    return pattern.format(w, w/tan(radians(ha)))
  def get_edges(self):
    '''
    Returns a plotable version of the indenter usable directly in ``matplotlib.pyplot``.
    
    :rtype: x and y lists
    '''
    from math import radians, tan
    pos = self.summit_position
    xs, ys = pos[0], pos[1]
    w = self.width
    ha = self.half_angle
    return [ w + xs, xs ], [ w / tan(radians(ha)) + ys, ys], [0., 0.]
    
  def apply_displacement(self, disp):
    '''
    Applies a displacement field to the indenter.
    
    :param disp: displacement field (with only one location).
    :type disp: ``abapy.postproc.VectorFieldOutput`` instance.
    '''
    from abapy.postproc import VectorFieldOutput
    if isinstance(disp, VectorFieldOutput) == False: raise Exception, 'disp must be VectorFieldOutput instance, got {0} instead'.format(type(disp))
    sp = self.summit_position
    new_position = ( sp[0] + disp.data1[0], sp[1] + disp.data2[0] )
    self.set_summit_position(new_position)


class DeformableCone2D:
  '''
  A deformable cone usable in 2D and Axisymmetric simulations.
  
  :param half_angle: half_angle in DEGREES.
  :type half_angle: float > 0.
  :param Na: number of elements along x axis in the finely meshed contact zone. *Must be power of 2*.
  :type Na: int
  :param Nb: number of elements along y axis in the finely meshed contact zone. *Must be power of 2*.
  :type Nb: int
  :param Ns: number of radial elements in the shell. 
  :type Ns: int
  :param Nf: number of orthoradial elements in each half shell. Must be > 0.
  :type Nf: int
  :param l: length of the square zone. 
  :type l: float.
  :param mat_label: label of the constitutive material of the indenter. 
  :type mat_label: any material class instance
  :param summit_position: position of the summit in a 2D space.
  :type summit_position: tuple or list containing two floats.
  :param rigid: True if indenter is to be rigid or False if the indenter is to be deformable. If the rigid behavior is chosen, the material label will be necessary but will not influence the results of the simulation.
  
  .. plot:: example_code/indentation/DeformableCone2D.py
      :include-source:    
  '''
  
  def __init__(self, half_angle= 70.3, Na = 4, Nb = 4, Ns = 4, Nf = 2, l = 1., mat_label = 'INDENTER_MAT', summit_position = (0., 0.), rigid = False):
    self.set_half_angle(half_angle)
    self.set_l(l)
    self.set_Na(Na)
    self.set_Nb(Nb)
    self.set_Ns(Ns)
    self.set_Nf(Nf)
    self.set_mat_label(mat_label)
    self.set_summit_position(summit_position)
    self.set_rigid(rigid)
    
    
  def set_summit_position(self, summit_position):
    '''
    Sets the position of the indenter.
    
    :param summit_position: position of the summit in a 2D space.
    :type summit_position: tuple or list containing two floats.
    '''
    if len(summit_position) != 2: raise Exception, 'summit_position length must be 2, got {0} instead.'.format(len(summit_position))
    self.summit_position = (float(summit_position[0]) , float(summit_position[1]) )
  
  def set_half_angle(self, half_angle = 70.3):
    '''
    Sets the half angle of the indenter. Default is equivalent to modified Berkovich indenter in volume.
    
    :param half_angle: half_angle in DEGREES.
    :type half_angle: float > 0.
    '''
    if type(half_angle) not in [float, int, long]:
      raise Exception, 'half angle must be float, got {0}'.format(type(half_angle))
    self.half_angle = abs(float(half_angle))
  def set_l(self,l):
    '''
    Sets the l parameter of the indenter (see ``ParamInfiniteMesh`` for explanations) 
    
    :param l: l
    :type l: float > 0.
    '''
    if type(l) not in [float, int, long]:
      raise Exception, 'l must be float, got {0}'.format(type(l))
    self.l = abs(float(l))
    
  def set_Na(self,Na):
    '''
    Sets the Na parameter of the indenter (see ``IndentationMesh`` for explanations).
    
    :param Na: Na
    :type Na: int > 1
    '''
    if type(Na) not in [int, long]:
      raise Exception, 'Na must be int > 1, got {0}'.format(type(Na))
    self.Na = abs(Na)
    
  def set_Nb(self,Nb):
    '''
    Sets the Nb parameter of the indenter (see ``IndentationMesh`` for explanations).
    
    :param Nb: Nb
    :type Nb: int > 1
    '''
    if type(Nb) not in [int, long]:
      raise Exception, 'Nb must be int > 1, got {0}'.format(type(Nb))
    self.Nb = abs(Nb)
  
  def set_Ns(self,Ns):
    '''
    Sets the Ns parameter of the indenter (see ```IndentationMesh`` for explanations).
    
    :param Ns: Ns
    :type Ns: int > 1
    '''
    if type(Ns) not in [int, long]:
      raise Exception, 'Ns must be int > 1, got {0}'.format(type(Ns))
    self.Ns = abs(Ns)
  
  def set_Nf(self,Nf):
    '''
    Sets the Nf parameter of the indenter (see ``IndentationMesh`` for explanations).
    
    :param Nf: Nf
    :type Nf: int > 1
    '''
    if type(Nf) not in [int, long]:
      raise Exception, 'Nf must be int > 1, got {0}'.format(type(Nf))
    self.Nf = abs(Nf)
  
  def set_mat_label(self,mat_label):
    '''
    Sets the label of the constitutive material of the indenter.
    
    :param mat_label: mat_label
    :type mat_label: string
    '''
    if type(mat_label) is not str:
      raise Exception, 'mat_label must be string, got {0}'.format(type(mat_label))
    self.mat_label = mat_label
  
  def set_rigid(self, rigid):
    '''
    Sets the indenter to be rigid (True) or deformable (False).
    
    :param rigid: True for rigid, False for deformable (default)
    :type rigid: bool
    '''
    if type(rigid) is not bool: raise Exception, 'rigid must be bool type, got {0} instead'.format(type(rigid))
    self.rigid = rigid
    
  
  def make_mesh(self):
    from numpy import tan, radians, pi, arcsin, sin, cos
    mesh = IndentationMesh(Na = self.Na, Nb = self.Nb, Ns = self.Ns, Nf = self.Nf, l = self.l, name = 'CAX4')
    mesh.add_set("INDENTER_ELEMENTS", mesh.labels)
    mesh = mesh.apply_reflection(point = (0., 0., 0.), normal = (0., 1., 0.) )
    ymax, xmax = max(mesh.nodes.y), max(mesh.nodes.x)
    #mesh.nodes.add_set_by_func('bottom', lambda x, y, z, labels: y == ymax)
    #mesh.nodes.add_set_by_func('reference', lambda x, y, z, labels: (x==0.) * (y==ymax) )
    #mesh.nodes.add_set('tip', 1)
    if self.rigid == False:
      rigid_nodes = mesh.nodes.sets['bottom_nodes']
      xsym_nodes = mesh.nodes.sets['axis_nodes']
    else:
      rigid_nodes = mesh.nodes.sets['all_nodes']
      xsym_nodes = mesh.nodes.sets['ref_node']
    mesh.nodes.add_set('rigid_nodes', rigid_nodes)
    mesh.nodes.add_set('xsym_nodes', xsym_nodes)
    mesh.surfaces = {}
    mesh.add_surface('surface_faces',[ ('surface_elements',1) ])
    '''
    def dispf( x, y, z, labels): 
      return 0.* x, x * (ymax-y)/ymax * tan(pi/2. - radians(self.half_angle)), 0.* x
    '''
    def dispf(x,y,z,labels):
      r = (x**2 + y**2)**.5
      r1 = r + 1. * (r == 0.) # Removing zeros
      theta = arcsin(x/r1) * radians(self.half_angle)/(pi/2.)
      
      x1 = r * sin(theta)
      y1 = r * cos(theta)
      return x1 - x, y1 -y, 0. * x
    disp = mesh.nodes.eval_vectorFunction(dispf)
    mesh.nodes.apply_displacement(disp)
    mesh.nodes.translate(x = self.summit_position[0], y = self.summit_position[1])
    self.mesh = mesh
      
  def dump2inp(self):
    '''
    Dumps to Abaqus INP format.
    
    :rtype: string
    '''
    self.make_mesh()
    out = self.mesh.dump2inp()
    out += '''
**----------------------------------
** ELEMENT SURFACES
**----------------------------------
*SOLID SECTION, ELSET = ALL_ELEMENTS, MATERIAL = {0}
*RIGID BODY, REF NODE = REF_NODE, PIN NSET=RIGID_NODES'''.format(self.mat_label.upper())
    
    return out
    
  def get_edges(self, **kwargs):
    '''
    Returns a plotable version of the indenter usable directly in ``matplotlib.pyplot``.
    
    :rtype: x and y lists
    '''
    return self.mesh.get_edges(kwargs)
  
  def get_border(self,**kwargs):
    '''
    Returns a plotable version of the border of the indenter usable directly in ``matplotlib.pyplot``.
    
    :rtype: x and y lists
    '''
    return self.mesh.get_border(kwargs)
    
  def apply_displacement(self, disp):
    '''
    Applies a displacement field to the indenter.
    
    :param disp: displacement field (with only one location).
    :type disp: ``abapy.postproc.VectorFieldOutput`` instance.
    '''
    self.mesh.nodes.apply_displacement(disp)
  def equivalent_half_angle(self):
    '''
    '''
    return self.half_angle
  

#-----------------------------------------------------------
# WORK IN PROGRESS
# Meta Indenter Class
class DeformableIndenter2D:
  '''
  A deformable indenter meta class usable in 2D and Axisymmetric simulations.
  
  :param half_angle: half_angle in DEGREES.
  :type half_angle: float > 0.
  :param Na: number of elements along x axis in the finely meshed contact zone. *Must be power of 2*.
  :type Na: int
  :param Nb: number of elements along y axis in the finely meshed contact zone. *Must be power of 2*.
  :type Nb: int
  :param Ns: number of radial elements in the shell. 
  :type Ns: int
  :param Nf: number of orthoradial elements in each half shell. Must be > 0.
  :type Nf: int
  :param l: length of the square zone. 
  :type l: float.
  :param mat_label: label of the constitutive material of the indenter. 
  :type mat_label: any material class instance
  :param summit_position: position of the summit in a 2D space.
  :type summit_position: tuple or list containing two floats.
  :param rigid: True if indenter is to be rigid or False if the indenter is to be deformable. If the rigid behavior is chosen, the material label will be necessary but will not influence the results of the simulation.
  
  .. plot:: example_code/indentation/DeformableCone2D.py
      :include-source:    
  '''
  
  def __init__(self, dispf, mesher, mat_label = 'INDENTER_MAT', summit_position = (0., 0.), rigid = False):
    self.set_dispf(dispf)
    self.set_mesher(mesher)
    self.set_mat_label(mat_label)
    self.set_summit_position(summit_position)
    self.set_rigid(rigid)
    
    
  def set_summit_position(self, summit_position):
    '''
    Sets the position of the indenter.
    
    :param summit_position: position of the summit in a 2D space.
    :type summit_position: tuple or list containing two floats.
    '''
    if len(summit_position) != 2: raise Exception, 'summit_position length must be 2, got {0} instead.'.format(len(summit_position))
    self.summit_position = (float(summit_position[0]) , float(summit_position[1]) )
  
  def set_half_angle(self, half_angle = 70.3):
    '''
    Sets the half angle of the indenter. Default is equivalent to modified Berkovich indenter in volume.
    
    :param half_angle: half_angle in DEGREES.
    :type half_angle: float > 0.
    '''
    if type(half_angle) not in [float, int, long]:
      raise Exception, 'half angle must be float, got {0}'.format(type(half_angle))
    self.half_angle = abs(float(half_angle))
  def set_l(self,l):
    '''
    Sets the l parameter of the indenter (see ``ParamInfiniteMesh`` for explanations) 
    
    :param l: l
    :type l: float > 0.
    '''
    if type(l) not in [float, int, long]:
      raise Exception, 'l must be float, got {0}'.format(type(l))
    self.l = abs(float(l))
    
  def set_Na(self,Na):
    '''
    Sets the Na parameter of the indenter (see ``IndentationMesh`` for explanations).
    
    :param Na: Na
    :type Na: int > 1
    '''
    if type(Na) not in [int, long]:
      raise Exception, 'Na must be int > 1, got {0}'.format(type(Na))
    self.Na = abs(Na)
    
  def set_Nb(self,Nb):
    '''
    Sets the Nb parameter of the indenter (see ``IndentationMesh`` for explanations).
    
    :param Nb: Nb
    :type Nb: int > 1
    '''
    if type(Nb) not in [int, long]:
      raise Exception, 'Nb must be int > 1, got {0}'.format(type(Nb))
    self.Nb = abs(Nb)
  
  def set_Ns(self,Ns):
    '''
    Sets the Ns parameter of the indenter (see ```IndentationMesh`` for explanations).
    
    :param Ns: Ns
    :type Ns: int > 1
    '''
    if type(Ns) not in [int, long]:
      raise Exception, 'Ns must be int > 1, got {0}'.format(type(Ns))
    self.Ns = abs(Ns)
  
  def set_Nf(self,Nf):
    '''
    Sets the Nf parameter of the indenter (see ``IndentationMesh`` for explanations).
    
    :param Nf: Nf
    :type Nf: int > 1
    '''
    if type(Nf) not in [int, long]:
      raise Exception, 'Nf must be int > 1, got {0}'.format(type(Nf))
    self.Nf = abs(Nf)
  
  def set_mat_label(self,mat_label):
    '''
    Sets the label of the constitutive material of the indenter.
    
    :param mat_label: mat_label
    :type mat_label: string
    '''
    if type(mat_label) is not str:
      raise Exception, 'mat_label must be string, got {0}'.format(type(mat_label))
    self.mat_label = mat_label
  
  def set_rigid(self, rigid):
    '''
    Sets the indenter to be rigid (True) or deformable (False).
    
    :param rigid: True for rigid, False for deformable (default)
    :type rigid: bool
    '''
    if type(rigid) is not bool: raise Exception, 'rigid must be bool type, got {0} instead'.format(type(rigid))
    self.rigid = rigid
    
  
  def make_mesh(self):
    from numpy import tan, radians, pi, arcsin, sin, cos
    mesh = IndentationMesh(Na = self.Na, Nb = self.Nb, Ns = self.Ns, Nf = self.Nf, l = self.l, name = 'CAX4')
    mesh = mesh.apply_reflection(point = (0., 0., 0.), normal = (0., 1., 0.) )
    ymax, xmax = max(mesh.nodes.y), max(mesh.nodes.x)
    #mesh.nodes.add_set_by_func('bottom', lambda x, y, z, labels: y == ymax)
    #mesh.nodes.add_set_by_func('reference', lambda x, y, z, labels: (x==0.) * (y==ymax) )
    #mesh.nodes.add_set('tip', 1)
    if self.rigid == False:
      rigid_nodes = mesh.nodes.sets['bottom_nodes']
      xsym_nodes = mesh.nodes.sets['axis_nodes']
    else:
      rigid_nodes = mesh.nodes.sets['all_nodes']
      xsym_nodes = mesh.nodes.sets['ref_node']
    mesh.nodes.add_set('rigid_nodes', rigid_nodes)
    mesh.nodes.add_set('xsym_nodes', xsym_nodes)
    mesh.surfaces = {}
    mesh.add_surface('surface_faces',[ ('surface_elements',1) ])
    '''
    def dispf( x, y, z, labels): 
      return 0.* x, x * (ymax-y)/ymax * tan(pi/2. - radians(self.half_angle)), 0.* x
    '''
    def dispf(x,y,z,labels):
      r = (x**2 + y**2)**.5
      r1 = r + 1. * (r == 0.) # Removing zeros
      theta = arcsin(x/r1) * radians(self.half_angle)/(pi/2.)
      
      x1 = r * sin(theta)
      y1 = r * cos(theta)
      return x1 - x, y1 -y, 0. * x
    disp = mesh.nodes.eval_vectorFunction(dispf)
    mesh.nodes.apply_displacement(disp)
    mesh.nodes.translate(x = self.summit_position[0], y = self.summit_position[1])
    self.mesh = mesh
      
  def dump2inp(self):
    '''
    Dumps to Abaqus INP format.
    
    :rtype: string
    '''
    self.make_mesh()
    out = self.mesh.dump2inp()
    out += '''
**----------------------------------
** ELEMENT SURFACES
**----------------------------------
*SOLID SECTION, ELSET = ALL_ELEMENTS, MATERIAL = {0}
*RIGID BODY, REF NODE = REF_NODE, PIN NSET=RIGID_NODES'''.format(self.mat_label.upper())
    
    return out
    
  def get_edges(self):
    '''
    Returns a plotable version of the indenter usable directly in ``matplotlib.pyplot``.
    
    :rtype: x and y lists
    '''
    return self.mesh.get_edges()
  
  def get_border(self):
    '''
    Returns a plotable version of the border of the indenter usable directly in ``matplotlib.pyplot``.
    
    :rtype: x and y lists
    '''
    return self.mesh.get_border()
    
  def apply_displacement(self, disp):
    '''
    Applies a displacement field to the indenter.
    
    :param disp: displacement field (with only one location).
    :type disp: ``abapy.postproc.VectorFieldOutput`` instance.
    '''
    self.mesh.nodes.apply_displacement(disp)
  def equivalent_half_angle(self):
    '''
    '''
    return self.half_angle

#-----------------------------------------------------------


  
class DeformableCone3D:
  '''
  A deformable cone usable in 3D simulations.
  
  :param half_angle: half_angle in DEGREES.
  :type half_angle: float > 0.
  :param Na: number of elements along x axis in the finely meshed contact zone. *Must be power of 2*.
  :type Na: int
  :param Nb: number of elements along y axis in the finely meshed contact zone. *Must be power of 2*.
  :type Nb: int
  :param Ns: number of radial elements in the shell. 
  :type Ns: int
  :param Nf: number of orthoradial elements in each half shell. Must be > 0.
  :type Nf: int
  :param l: length of the square zone. 
  :type l: float.
  :param N: Number of sweeped elements. 
  :type N: int.
  :param sweep_angle: sweep angle. 
  :type l: float.
  :param mat_label: label of the constitutive material of the indenter. 
  :type mat_label: any material class instance
  :param summit_position: position of the summit in a 2D space.
  :type summit_position: tuple or list containing two floats.
  :param rigid: True if indenter is to be rigid or False if the indenter is to be deformable. If the rigid behavior is chosen, the material label will be necessary but will not influence the results of the simulation.
  :param pyramid:  Sets the indenter as a revolution cone (False) or a pyramid (True). I the case of the pyramid, the half angle becomes the axis to face angle.
  :type pyramid: bool
  
  For common 3D indenters, following parameters can be used:
     
  +---------------------+------------------+------------------+
  | Indenter            |     half_angle   |     sweep_angle  |
  +=====================+==================+==================+
  | Berkovich           |       65.03      |      60.00       |
  +---------------------+------------------+------------------+
  | Modified Berkovich  |       65.27      |      60.00       |
  +---------------------+------------------+------------------+
  | Cube Corner         |       35.26      |      60.00       |
  +---------------------+------------------+------------------+
  | Vickers             |       68.00      |      45.00       |
  +---------------------+------------------+------------------+
  
  .. plot:: example_code/indentation/DeformableCone3D.py
      :include-source:    
      
  
  '''
  
  def __init__(self, half_angle= 70.3, Na = 4, Nb = 4, Ns = 4, Nf = 2, l = 1., N = 4, sweep_angle=45., mat_label = 'INDENTER_MAT', summit_position = (0., 0.), rigid = True, pyramid = False):
    self.set_half_angle(half_angle)
    self.set_l(l)
    self.set_Na(Na)
    self.set_Nb(Nb)
    self.set_Ns(Ns)
    self.set_Nf(Nf)
    self.set_N(N)
    self.set_sweep_angle(sweep_angle)
    self.set_mat_label(mat_label)
    self.set_summit_position(summit_position)
    self.set_rigid(rigid)
    self.set_pyramid(pyramid)
    
    
  def set_summit_position(self, summit_position):
    '''
    Sets the position of the indenter.
    
    :param summit_position: position of the summit in a 2D space.
    :type summit_position: tuple or list containing two floats.
    '''
    if len(summit_position) != 2: raise Exception, 'summit_position length must be 2, got {0} instead.'.format(len(summit_position))
    self.summit_position = (float(summit_position[0]) , float(summit_position[1]) )
  
  def set_half_angle(self, half_angle = 70.3):
    '''
    Sets the half angle of the indenter. Default is equivalent to modified Berkovich indenter in volume.
    
    :param half_angle: half_angle in DEGREES.
    :type half_angle: float > 0.
    '''
    if type(half_angle) not in [float, int, long]:
      raise Exception, 'half angle must be float, got {0}'.format(type(half_angle))
    self.half_angle = abs(float(half_angle))
  def set_l(self,l):
    '''
    Sets the l parameter of the indenter (see ``ParamInfiniteMesh`` for explanations) 
    
    :param l: l
    :type l: float > 0.
    '''
    if type(l) not in [float, int, long]:
      raise Exception, 'l must be float, got {0}'.format(type(l))
    self.l = abs(float(l))
    
  def set_Na(self,Na):
    '''
    Sets the Na parameter of the indenter (see ``IndentationMesh`` for explanations).
    
    :param Na: Na
    :type Na: int > 1
    '''
    if type(Na) not in [int, long]:
      raise Exception, 'Na must be int > 1, got {0}'.format(type(Na))
    self.Na = abs(Na)
    
  def set_Nb(self,Nb):
    '''
    Sets the Nb parameter of the indenter (see ``IndentationMesh`` for explanations).
    
    :param Nb: Nb
    :type Nb: int > 1
    '''
    if type(Nb) not in [int, long]:
      raise Exception, 'Nb must be int > 1, got {0}'.format(type(Nb))
    self.Nb = abs(Nb)
  
  def set_Ns(self,Ns):
    '''
    Sets the Ns parameter of the indenter (see ```IndentationMesh`` for explanations).
    
    :param Ns: Ns
    :type Ns: int > 1
    '''
    if type(Ns) not in [int, long]:
      raise Exception, 'Ns must be int > 1, got {0}'.format(type(Ns))
    self.Ns = abs(Ns)
  
  def set_Nf(self,Nf):
    '''
    Sets the Nf parameter of the indenter (see ``IndentationMesh`` for explanations).
    
    :param Nf: Nf
    :type Nf: int > 1
    '''
    if type(Nf) not in [int, long]:
      raise Exception, 'Nf must be int > 1, got {0}'.format(type(Nf))
    self.Nf = abs(Nf)
  
  def set_N(self,N):
    '''
    Sets the number of sweeped elements
    
    :param N: N
    :type N: int > 1
    '''
    if type(N) not in [int, long]:
      raise Exception, 'N must be int > 0, got {0}'.format(type(N))
    self.N = abs(N)
  
  def set_sweep_angle(self,sweep_angle):
    '''
    Sets the sweep angle.
    
    :param sweep_angle: sweep_angle
    :type sweep_angle: int > 1
    '''
    if type(sweep_angle) != float:
      raise Exception, 'sweep_angle must be float, got {0}'.format(type(sweep_angle))
    self.sweep_angle = sweep_angle
  
  def set_mat_label(self,mat_label):
    '''
    Sets the label of the constitutive material of the indenter.
    
    :param mat_label: mat_label
    :type mat_label: string
    '''
    if type(mat_label) is not str:
      raise Exception, 'mat_label must be string, got {0}'.format(type(mat_label))
    self.mat_label = mat_label
  
  def set_rigid(self, rigid):
    '''
    Sets the indenter to be rigid (True) or deformable (False).
    
    :param rigid: True for rigid, False for deformable (default)
    :type rigid: bool
    '''
    if type(rigid) is not bool: raise Exception, 'rigid must be bool type, got {0} instead'.format(type(rigid))
    self.rigid = rigid
  
  def set_pyramid(self, pyramid):
    '''
    Sets the indenter as a revolution cone (False) or a pyramid (True). I the case of the pyramid, the half angle becomes the axis to face angle.
    
    :param pyramid: True for pyramid, False for revolution (default).
    :type pyramid: bool
    '''
    if type(pyramid) is not bool: raise Exception, 'pyramid must be bool type, got {0} instead'.format(type(pyramid))
    self.pyramid = pyramid  
  
  def equivalent_half_angle(self):
    '''
    Returns the half angle (in degrees) of the equivalent cone in terms of cross area and volume.
    :retype: float
    '''
    if self.pyramid == False:
      return self.half_angle 
    else:
      import numpy as np
      psi = np.radians(self.half_angle)
      alpha = np.radians(self.sweep_angle)
      Ap = 2 * np.pi / alpha * np.tan(psi)**2 *np.tan(alpha) / 2.
      rc = ( Ap/np.pi )**.5
      phi = np.arctan(rc)
      return np.degrees(phi)
  
  def make_mesh(self):
    from numpy import tan, radians, pi, arcsin, sin, cos
    import copy
    mesh = IndentationMesh(Na = self.Na, Nb = self.Nb, Ns = self.Ns, Nf = self.Nf, l = self.l, name = 'CAX4')
    mesh = mesh.apply_reflection(point = (0., 0., 0.), normal = (0., 1., 0.) )
    def dispf(x,y,z,labels):
      r = (x**2 + y**2)**.5
      r1 = r + 1. * (r == 0.) # Removing zeros
      theta = arcsin(x/r1) * radians(self.half_angle)/(pi/2.)
      x1 = r * sin(theta)
      y1 = r * cos(theta)
      return x1 - x, y1 -y, 0. * x
    disp = mesh.nodes.eval_vectorFunction(dispf)
    mesh.nodes.apply_displacement(disp)
    mesh.nodes.translate(x = self.summit_position[0], y = self.summit_position[1])
    mesh.surfaces = {}
    mesh.add_surface('surface_faces',[ ('surface_elements',1) ])
    mesh = mesh.sweep(N=self.N, sweep_angle = self.sweep_angle, extrude = self.pyramid)
    # Removing some sets intersections
    mesh.nodes.sets['all_front_nodes'] = copy.copy(mesh.nodes.sets['front_nodes'])
    mesh.nodes.sets['all_back_nodes'] = copy.copy(mesh.nodes.sets['back_nodes'])
    front_nodes = mesh.nodes.sets['front_nodes']
    back_nodes = mesh.nodes.sets['back_nodes']
    axis_nodes = mesh.nodes.sets['axis_nodes']
    bottom_nodes = mesh.nodes.sets['bottom_nodes']
    def remove_intersection(target):
      inter0 = set(target) & set(axis_nodes)
      inter1 = set(target) & set(bottom_nodes)
      inter = list(inter0 | inter1)
      for n in inter: target.remove(n)
      
    remove_intersection(back_nodes)
    remove_intersection(front_nodes)
    if self.rigid == False:
      rigid_nodes = mesh.nodes.sets['bottom_nodes']
      xsym_nodes = mesh.nodes.sets['axis_nodes']
    else:
      rigid_nodes = mesh.nodes.sets['all_nodes']
      xsym_nodes = mesh.nodes.sets['ref_node']
    mesh.nodes.add_set('rigid_nodes', rigid_nodes)
    mesh.nodes.add_set('xsym_nodes', xsym_nodes)
    
    '''
    def dispf( x, y, z, labels): 
      return 0.* x, x * (ymax-y)/ymax * tan(pi/2. - radians(self.half_angle)), 0.* x
    '''
    
    self.mesh = mesh
      
  def dump2inp(self):
    '''
    Dumps to Abaqus INP format.
    
    :rtype: string
    '''
    self.make_mesh()
    out = self.mesh.dump2inp()
    if self.sweep_angle != 360. : 
      out += '''
**----------------------------------
** CSYS CHANGE TO CYLINDRICAL
**----------------------------------
*TRANSFORM, TYPE=C, NSET=FRONT_NODES
 0., 0., 0., 0., 1.,0.    
*TRANSFORM, TYPE=C, NSET=BACK_NODES
 0., 0., 0., 0., 1.,0.'''    
    out += '''
**----------------------------------
** SOLID SECTION
**----------------------------------
*SOLID SECTION, ELSET = ALL_ELEMENTS, MATERIAL = {0}
**----------------------------------
** RIGID BODY
**----------------------------------
*RIGID BODY, REF NODE = REF_NODE, PIN NSET=RIGID_NODES'''.format(self.mat_label.upper())
    
    return out
    
  def get_edges(self):
    '''
    Returns a plotable version of the indenter usable directly in ``matplotlib.pyplot``.
    
    :rtype: x and y lists
    '''
    return self.mesh.get_edges()
  
  def get_border(self):
    '''
    Returns a plotable version of the border of the indenter usable directly in ``matplotlib.pyplot``.
    
    :rtype: x and y lists
    '''
    return self.mesh.get_border()
    
  def apply_displacement(self, disp):
    '''
    Applies a displacement field to the indenter.
    
    :param disp: displacement field (with only one location).
    :type disp: ``abapy.postproc.VectorFieldOutput`` instance.
    '''
    self.mesh.nodes.apply_displacement(disp)
  
class Step(object):
  '''
  Builds a typical indentation step.
  
  :param name: step name.
  :type name: string
  :param disp: displacement.
  :type disp: float > 0.
  :param nframes: frame number.
  :type nframes: int
  :param nlgeom: nlgeom state.
  :type nlgeom: boolean
  :param fieldOutputFreq: field output frequency
  :type fieldOutputFreq: int
  :param boundaries_3D: 3D or 2D boundary conditions. If 3D is True, then boundary conditions will be applied to the node sets ``front`` and ``back``.
  :type boundaries_3D: boolean 
  :param full_3D: set to True if the model is a complete 3D model without symmetries and then does not need side boundaries.
  :type full_3D: boolean
  :param rigid_indenter_3D: Set to True if a 3D indenter is rigid  
  :type rigid_indenter_3D: boolean
  :param nodeFieldOutput: node outputs.
  :type nodeFieldOutput: string or list of strings
  :param elemFieldOutput: node outputs.
  :type elemFieldOutput: string or list of strings
  
  '''
  
  def __init__(self, name, disp = 1., nlgeom = True, nframes = 100, fieldOutputFreq = 999999, boundaries_3D = False, full_3D = False, rigid_indenter_3D = True,  nodeFieldOutput = ['COORD', 'U'], elemFieldOutput = ['LE', 'EE', 'PE', 'PEEQ', 'S'], mode = 'bulk'):
    self.set_name(name)
    self.set_displacement(disp)
    self.set_nframes(nframes)
    self.set_nlgeom(nlgeom)
    self.set_fieldOutputFreq(fieldOutputFreq)
    self.set_boundaries_3D(boundaries_3D)
    self.set_rigid_indenter_3D(rigid_indenter_3D)
    self.set_full_3D(full_3D)
    self.set_nodeFieldOutput(nodeFieldOutput)
    self.set_elemFieldOutput(elemFieldOutput)
    modes = ['bulk', 'film', 'film_substrate']
    if mode not in modes: raise Exception('mode must be in {0}'.format(modes))
    self.mode = mode
    
  def set_name(self,name):
    '''
    Sets step name.
    
    :param name: step name.
    :type name: string
    '''
    if type(name) is not str: raise Exception, 'name must be str, got {0}'.format(type(anme))
    self.name = name
  def set_displacement(self,disp):
    '''
    Sets the displacement.
    
    :param disp: displacement.
    :type disp: float > 0.
    '''
    if type(disp) not in [float, int, long]: raise Exception, 'disp must be float, got {0}'.format(type(disp))
    self.disp = float(disp)
  def set_nframes(self, nframes):
    '''
    Sets the number of frames.
    
    :param nframes: frame number.
    :type nframes: int
    '''
    if type(nframes) not in [int, long]: raise Exception, 'disp must be int > 0, got {0}'.format(type(nframes))
    self.nframes = nframes
  def set_nlgeom(self, nlgeom):
    '''
    Sets NLGEOM on or off.
    
    :param nlgeom: nlgeom state.
    :type nlgeom: boolean
    '''
    if type(nlgeom) is not bool: raise Exception, 'nlgeom must be boolean, got {0}'.format(type(nlgeom))
    self.nlgeom = nlgeom
  def set_fieldOutputFreq(self, freq):
    '''
    Sets the field output period.
    
    :param freq: field output frequency
    :type freq: int
    '''
    if type(freq) not in [int, long]: raise Exception, 'freq must be int, got {0}'.format(type(freq))
    self.fieldOutputFreq = freq
    
  def set_boundaries_3D(self, boundaries_3D):
    '''
    Sets the 3D mode for step.
    
    :param boundaries_3D: 3D or 2D boundary conditions. If 3D is True, then boundary conditions will be applied to the node sets ``front`` and ``back``.
  :type boundaries_3D: boolean 
    '''
    if type(boundaries_3D) != bool : raise Exception, 'boundaries_3D must be bool, got {0}'.format(type(boundaries_3D))
    self.boundaries_3D = boundaries_3D
    
  def set_full_3D(self, full_3D):
    '''
    Sets the 3D mode for step.
    
    :param full_3D: set to True if the model is a complete 3D model without symmetries and then does not need side boundaries.
  :type full_3D: boolean 
    '''
    if type(full_3D) != bool : raise Exception, 'full_3D must be bool, got {0}'.format(type(full_3D))
    self.full_3D = full_3D  
  
  def set_rigid_indenter_3D(self, rigid_indenter_3D):
    '''
    Sets the 3D rigid indenter.
    
    :param rigid_indenter_3D: Set to True if a 3D indenter is rigid  
  :type rigid_indenter_3D: boolean
    '''
    if type(rigid_indenter_3D) != bool : raise Exception, 'rigid_indenter_3D must be bool, got {0}'.format(type(rigid_indenter_3D))
    self.rigid_indenter_3D = rigid_indenter_3D
    
  def set_nodeFieldOutput(self, nodeOutput):
    '''
    Sets the node field output to be recorded.
    
    :param nodeOutput: node outputs.
    :type nodeOutput: string or list of strings
    '''
    if type(nodeOutput) is str:
      self.nodeFieldOutput = [nodeOutput]
    else:
      for no in nodeOutput:
        if type(no) is not str: raise Exception, 'node outputs must be strings, got {0}'.format(type(no))
      self.nodeFieldOutput = nodeOutput
  def set_elemFieldOutput(self, elemOutput):
    '''
    Sets the element field output to be recorded.
    
    :param elemOutput: node outputs.
    :type elemOutput: string or list of strings
    '''
    if type(elemOutput) is str:
      self.elemFieldOutput = [nodeOutput]
    else:
      for eo in elemOutput:
        if type(eo) is not str: raise Exception, 'element outputs must be strings, got {0}'.format(type(no))
      self.elemFieldOutput = elemOutput
  def dump2inp(self):
    '''
    Dumps the step to Abaqus INP format.
    '''
    stepPattern = '''*STEP, NAME = {0}, NLGEOM = {1}, INC=1000000
*STATIC, DIRECT
{2}, 1.
*BOUNDARY
I_SAMPLE.AXIS_NODES, 1, 1
I_SAMPLE.BOTTOM_NODES, 1, 3{7}
I_INDENTER.XSYM_NODES, 1, 1
I_INDENTER.XSYM_NODES, 3, 3
I_INDENTER.REF_NODE, 4, 6 
I_INDENTER.REF_NODE, 2, 2, {3}
*RESTART, WRITE, FREQUENCY = 0
*OUTPUT, FIELD, FREQUENCY = {4}
*NODE OUTPUT
{5}
*NODE OUTPUT, NSET=I_INDENTER.REF_NODE
U
*ELEMENT OUTPUT, ELSET=I_SAMPLE.ALL_ELEMENTS, DIRECTIONS = YES
{6}
*ELEMENT OUTPUT, ELSET=I_INDENTER.ALL_ELEMENTS, DIRECTIONS = YES
{6}
*OUTPUT, HISTORY
*ENERGY OUTPUT
ALLFD, ALLWK
*ENERGY OUTPUT, ELSET=I_SAMPLE.ALL_ELEMENTS
ALLPD, ALLSE
*ENERGY OUTPUT, ELSET=I_INDENTER.INDENTER_ELEMENTS
ALLPD, ALLSE
*CONTACT OUTPUT
CAREA
*CONTACT OUTPUT, NSET=I_SAMPLE.TOP_NODES
CPRESS
*NODE OUTPUT, NSET=I_INDENTER.REF_NODE
U2
*NODE OUTPUT, NSET=I_INDENTER.TIP_NODE
U2
*NODE OUTPUT, NSET=I_INDENTER.REF_NODE
RF2
*NODE OUTPUT, NSET = I_SAMPLE.TOP_NODES
COOR1, COOR2, COOR3
*END STEP
'''
    # NLGEOM
    if self.nlgeom == True: 
      nlgeom = 'YES'
    else:
      nlgeom = 'NO'
      
    # BOUNDARY CONDITIONS
    bc_3D = ''
    if self.boundaries_3D:
      if self.full_3D == False:
        bc_3D += '\nI_SAMPLE.FRONT_NODES, 2, 2\nI_SAMPLE.BACK_NODES, 2, 2\nI_SAMPLE.AXIS_NODES, 3, 3\nI_INDENTER.REF_NODE, 3, 6'
      else:
        bc_3D += '\nI_SAMPLE.AXIS_NODES, 3, 3\nI_INDENTER.REF_NODE, 3, 6'  
      if self.rigid_indenter_3D == False:
        bc_3D += '\nI_INDENTER.FRONT_NODES, 2, 2\nI_INDENTER.BACK_NODES, 2, 2\nI_INDENTER.AXIS_NODES, 1, 1\nI_INDENTER.AXIS_NODES, 3, 3'
      
    # FIELD OUTPUTS  
    efos = ''
    for fo in self.elemFieldOutput:
      efos += '{0}, '.format(fo)   
    nfos = ''
    for fo in self.nodeFieldOutput:
      nfos += '{0}, '.format(fo)      
    out =  stepPattern.format(self.name.upper(), nlgeom, 1./self.nframes, -self.disp, self.fieldOutputFreq, nfos, efos, bc_3D) 
    return out
    
    
def MakeInp(sample_mesh = None, indenter = None, sample_mat = None, indenter_mat = None, friction = 0.0, steps = None, is_3D = False, heading = 'Abapy Indentation Simulation'):
  '''
  Builds a complete indentation INP for Abaqus and returns it as a string.
  
  :param sample_mesh: mesh to use for the sample. If None, default ``ParamInfiniteMesh`` will be used.
  :type sample_mesh: ``abapy.mesh.Mesh`` instance or None 
  :param indenter: indenter to use. If None, default ``RigidCone2D`` will be used.
  :type indenter: any indenter instance or None
  :param sample_mat: sample material to use. If None, default ``abapy.materials.VonMises`` will be used.
  :type sample_mat: any material instance or None
  :param indenter_mat: indenter material to use. If None, a default elastic material will be used. If a rigid indenter is chosen, this material will not interfer with the simulation results.
  :type indenter_mat: any material instance or None
  :param friction: friction coefficient between indenter and sample.
  :type friction: float >= 0.
  :param steps: steps to used during the test.
  :type steps: list of ``Step`` instances or None
  :rtype: string
  :type is_3D: set to True if the model is 3D, it will allow needed CSYS changes to cylindrical to be done.
  :type 3D: bool
  
  
  .. literalinclude:: example_code/indentation/MakeInp.py
  
  Returns: 
  
  :download:`indentation_axi.inp <example_code/indentation/workdir/indentation_axi.inp>`
  
  :download:`indentation_berko.inp <example_code/indentation/workdir/indentation_berko.inp>`
  '''
  import copy

  introPattern = '''**----------------------------------
**INDENTATION SIMULATION
**----------------------------------
*HEADING
{6}
*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO, CONTACT=NO
**----------------------------------
** SAMPLE DEFINITION
*PART, NAME = P_SAMPLE
{0}
*SOLID SECTION, ELSET = ALL_ELEMENTS, MATERIAL = SAMPLE_MAT{5}
*END PART
**----------------------------------
** INDENTER DEFINITION
**----------------------------------
*PART, NAME = P_INDENTER
{1}
*END PART
**----------------------------------
** ASSEMBLY
**----------------------------------
*ASSEMBLY, NAME = ASSEMBLY
*INSTANCE, NAME = I_SAMPLE, PART = P_SAMPLE
*END INSTANCE
*INSTANCE, NAME = I_INDENTER, PART= P_INDENTER
*END INSTANCE
*END ASSEMBLY
**----------------------------------
** SURFACE INTERACTIONS
**----------------------------------
*SURFACE INTERACTION, NAME = SURF_INT
*FRICTION
{2},
*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE = HARD
*CONTACT PAIR, INTERACTION = SURF_INT, SUPPLEMENTARY CONSTRAINTS = NO, TYPE= NODE TO SURFACE
I_SAMPLE.SURFACE_FACES, I_INDENTER.SURFACE_FACES
**----------------------------------
** MATERIALS
**----------------------------------
** SAMPLE MATERIAL
{3}
** INDENTER MATERIAL
{4}
**----------------------------------
** STEPS
**----------------------------------
'''
  if sample_mesh == None:
    sample_mesh =  ParamInfiniteMesh()
  if indenter == None:
    indenter = RigidCone2D()
  if sample_mat == None:
    from materials import VonMises
    sample_mat = VonMises(labels = 'SAMPLE_MAT')
  if indenter_mat == None:
    from materials import Elastic
    indenter_mat = Elastic(labels = 'INDENTER_MAT')
  if is_3D:
    csys = '''\n*TRANSFORM, TYPE=C, NSET=FRONT_NODES
0., 0., 0., 0., 1.,0.
*TRANSFORM, TYPE=C, NSET=BACK_NODES
0., 0., 0., 0., 1.,0.'''
    # Removing some sets intersections
    sample_mesh.nodes.sets['all_front_nodes'] = copy.copy(sample_mesh.nodes.sets['front_nodes'])
    sample_mesh.nodes.sets['all_back_nodes'] = copy.copy(sample_mesh.nodes.sets['back_nodes'])
    front_nodes = sample_mesh.nodes.sets['front_nodes']
    back_nodes = sample_mesh.nodes.sets['back_nodes']
    axis_nodes = sample_mesh.nodes.sets['axis_nodes']
    bottom_nodes = sample_mesh.nodes.sets['bottom_nodes']
    def remove_intersection(target):
      inter0 = set(target) & set(axis_nodes)
      inter1 = set(target) & set(bottom_nodes)
      inter = list(inter0 | inter1)
      for n in inter: target.remove(n)
      
    remove_intersection(back_nodes)
    remove_intersection(front_nodes)
  else:
    csys = ''
  if steps == None:
    steps = [Step(name = 'LOADING', disp = 1.), Step(name = 'UNLOADING', disp = 0.)]
  if isinstance(steps, Step): steps = [steps]
  sample_mat.labels = ['SAMPLE_MAT']
  indenter_mat.labels = ['INDENTER_MAT']
  out = introPattern.format(sample_mesh.dump2inp(), indenter.dump2inp(), friction , sample_mat.dump2inp(), indenter_mat.dump2inp(), csys, heading)
  for step in steps:
    out += step.dump2inp()
  
    
  return out


class Manager:
  '''
  The spirit of the simulation manager is to allow you to work at a higher level. Using it allows you to define once and for all where you work, where Abaqus is, it fill manage subprocesses (Abaqus, Abaqus python) automatically. It is particularly interresting to perform parametric simulation because you can modify one parameter (material property, inenter property, ...) and keep all other parameters fixed and rerun directly the simulation process without making any mistake.
  
  .. note:: This class is still under developpement, important changes may then happen.
    
  :param workdir: work directory where simulation is to be run.
  :type workdir: string
  :param abqlauncher: abaqus launcher or path to it. Take care about aliases under linux because they often don't work under non interactive shells. A direct path to the launcher may be a good idea.
  :type abqlauncher: string
  :param samplemesh: mesh to be used by ``MakeInp``, None will let ``MakeInp`` use it's own default.
  :type samplemesh: ``abapy.mesh.Mesh`` instance or None
  :param indenter: indenter to be used by ``MakeInp``, None will let ``MakeInp`` use it's own default.
  :type indenter: indenter instance or None
  :param samplemat: sample material to be used by ``MakeInp``, None will let ``MakeInp`` use it's own default.
  :type samplemat: material instance or None
  :param indentermat: indenter material to be used by ``MakeInp``, None will let ``MakeInp`` use it's own default.
  :type indentermat: material instance or None
  :param steps: steps to use during the test.
  :type steps: list of ``Step`` instances.
  :param is_3D: has to be True if the simulation is 3D, else must be False.
  :type is_3D: Bool
  :param simname: simulation name used for all files.
  :type simname: string
  :param files2delete: file types to delete when cleaning work directory.
  :type files2delete: list of strings.
 
  .. literalinclude:: example_code/indentation/Manager.py
  
  Gives:
  
  .. image:: example_code/indentation/workdir/indentation_field.png
  .. image:: example_code/indentation/workdir/indentation_load-disp.png
  
  .. note:: In order to used abaqus Python, you have to build a post processing script that is executed in ``abaqus python``. Here is an example :download:`abqpostproc.py <example_code/indentation/workdir/abqpostproc.py>`:
  
  .. literalinclude:: example_code/indentation/workdir/abqpostproc.py
  
  '''
  def __init__(self, workdir = '', abqlauncher = 'abaqus', samplemesh = None, indenter = None, samplemat = None, indentermat = None, friction = 0. ,steps = None, is_3D = False, simname = 'indentation', files2delete = ['sta', 'sim', 'prt', 'odb', 'msg', 'log', 'dat', 'com', 'inp','lck','pckl'], abqpostproc = 'abqpostproc.py'):
    from materials import Elastic, VonMises
    self.set_workdir(workdir)
    self.set_abqlauncher(abqlauncher)
    if samplemesh == None: samplemesh = ParamInfiniteMesh()
    self.set_samplemesh(samplemesh)
    if indenter == None: indenter = RigidCone2D()
    self.set_indenter(indenter)
    if samplemat == None:
      from abapy.materials import VonMises
      samplemat = VonMises()
    if indentermat == None:
      from abapy.materials import Elastic
      indentermat = Elastic()
    self.set_samplemat(samplemat)
    self.set_indentermat(indentermat)
    self.set_friction(friction)
    if steps == None: 
      steps = [Step(name = 'loading', disp = 1.),Step(name = 'unloading', disp = 0.) ]
    self.set_steps(steps)
    self.set_is_3D(is_3D)
    self.set_simname(simname)
    self.set_files2delete(files2delete)
    self.set_abqpostproc(abqpostproc)
    
    
    
  def set_workdir(self, workdir):
    '''
    Sets work directory
    
    :param workdir: relative or absolute path to workdir where simulations are run.
    :type workdir: string
    '''
    if type(workdir) is not str: raise Exception, 'workdir must be string, got {0} instead.'.format(type(workdir))
    self.workdir =  workdir
  def set_abqlauncher(self,abqlauncher):
    '''
    Sets Abaqus launcher
    :param abqlauncher: alias, relative path or absolute path to abaqus launcher. 
    :type abqlaucher: string
    
    .. note: aliases may not work because they are often limited to interactive shells.
    '''
    if type(abqlauncher) is not str: raise Exception, 'workdir must be string, got {0} instead.'.format(type(abqlauncher))
    self.abqlauncher =  abqlauncher
  def set_samplemesh(self, samplemesh):
    '''
    Sets sample mesh.
    
    :param samplemesh: sample mesh.
    :type samplemesh: ``abapy.mesh.Mesh`` instance
    '''
    from abapy.mesh import Mesh
    if isinstance(samplemesh, Mesh) == False: raise Exception, 'samplemesh must be Mesh instance, got {0} instead.'.format(type(samplemesh))
    self.samplemesh = samplemesh
  def set_indenter(self, indenter):
    '''
    Sets indenter.
    
    :param indenter: indenter to be used.
    :type indenter: instance of any indenter class
    '''
    self.indenter = indenter
  def set_samplemat(self, samplemat):
    '''
    Sets sample material.
    
    :param samplemat: core material.
    :type samplemat: instance of any material class
    '''
    samplemat.labels = ['SAMPLE_MAT']
    self.samplemat = samplemat
  
  def set_indentermat(self, indentermat):
    '''
    Sets indenter material.
    
    :param indentermat: indenter material.
    :type indentermat: instance of any material class
    '''
    indentermat.labels = ['SAMPLE_MAT']
    self.indentermat = indentermat
 
  def set_friction(self, friction):
    '''
    Sets the friction coefficient.
    
    :param friction: friction coefficient.
    :type friction: float >= 0.
    '''
    self.friction = friction
    
  def set_steps(self, steps):
    '''
    Sets steps
    
    :param steps: description of steps.
    :type steps: list ``of Steps`` instances
    '''
    for step in steps:
      if isinstance(step, Step) == False: raise Exception, 'step must be Step instance, got {0} instead.'.format(type(step))
    self.steps = steps
  
  def set_is_3D(self, is_3D):
    '''
    Sets the 3D flag as True or False.
    
    :param is_3D: has to be True if the simulation is 3D, else must be False.
    :type is_3D: Bool
    '''
    self.is_3D = is_3D
  
  def set_simname(self, simname):
    '''
    Sets simname.
    
    :param simname: simulation name that is used to name simulation files.
    :type simname: string
    '''
    if type(simname) is not str: raise Exception, 'simname must be string, got {0} instead.'.format(type(simname))
    self.simname = simname
  def set_files2delete(self,files2delete):
    '''
    Sets files to delete when cleaning.
    
    :param files2delete: files types to be deleted when cleaning workdir.
    :type files2delete: list of strings.
    '''
    for f in files2delete:
      if type(f) is not str: raise Exception, 'file must be string, got {0} instead'.format(type(f))
    self.files2delete = files2delete
  def make_inp(self):
    '''
    Builds the INP file using ``MakeInp`` and stores it in workdir as "simname.inp".
    '''
    out = MakeInp(sample_mesh = self.samplemesh, indenter = self.indenter, sample_mat = self.samplemat, indenter_mat = self.indentermat, friction = self.friction, steps = self.steps, is_3D = self.is_3D)
    pattern = '{0}{1}.inp'.format(self.workdir,self.simname )
    print '< Creating INP file: {0} >'.format(pattern)
    f = open(pattern, 'w')
    f.write(out)
    f.close()
  def run_sim(self):
    '''
    Runs the simulation.
    '''
    
    import os, time, subprocess
    t0 = time.time()
    print '< Running simulation {0} in Abaqus>'.format(self.simname)  
    command = '{0} job={1} input={1}.inp interactive'.format(self.abqlauncher, self.simname) 
    print command
    p = subprocess.Popen(command, cwd = self.workdir, shell=True, stdout = subprocess.PIPE)
         
    # Case for using UMAT subroutines
    from abapy.materials import SiDoLo
    print '============='
    print isinstance(self.samplemat,SiDoLo)
    print '============='
    if isinstance(self.samplemat,SiDoLo): 
      self.abqlauncher = '/vol/app/Abaqus-dev64/6.10-1/exec/abq6101.exe'
      command = '{0} job={1} input={1}.inp user={2} interactive'.format(self.abqlauncher, self.simname, self.samplemat.umat[0])
      print command
      p = subprocess.Popen(command, cwd = self.workdir, shell=True, stdout = subprocess.PIPE)
    trash = p.communicate()
    t1 = time.time()
    self.duration = t1 - t0
    print '< Ran {0} in Abaqus: duration {1:.2f}s>'.format(self.simname, t1 - t0)   
  
  def __repr__(self): 
    return '<abapy.indentation.Manager instance>'  
  
  def erase_files(self):
    '''
    Erases all files with types declared in files2delete in the work directory with the name *simname*.
    '''
    import os
    print '< Removing temporary files>'
    pattern = 'rm -f {0}{1}.{2}'
    for f in self.files2delete:
      os.system( pattern.format(self.workdir, self.simname, f) )
    
  def set_abqpostproc(self, abqpostproc):
    '''
    Sets the path to the abaqus post-processing python script, this path must be absolute or relative to workdir.
    
    :param abqpostproc: link to the abaqus post processing script.
    :type abqpostproc: string
    '''
    if type(abqpostproc) is not str: raise Exception, 'abqpostproc must be string'
    self.abqpostproc = abqpostproc
  
  def run_abqpostproc(self):
    '''
    Runs the first pass of post processing inside Abaqus Python.
    '''
    import os, subprocess, time
    t0 = time.time()
    #p = subprocess.Popen( [self.abqlauncher, 'python' ,self.abqpostproc, self.simname], cwd = self.workdir,stdout = subprocess.PIPE )
    p = subprocess.Popen( [self.abqlauncher,  'viewer', 'noGUI={0}'.format(self.abqpostproc)], cwd = self.workdir,stdout = subprocess.PIPE )
    trash = p.communicate()
    t1 = time.time()
    print '< Post Processed {0} in Abaqus: duration {1:.2f}s>'.format(self.simname, t1 - t0)   
    
    
  def set_pypostprocfunc(self, func):
    '''
    Sets the Python post processing function.
    
    :param func: post processing function of the data produced by abaqus post processing.
    :type func: function
    '''
    self.pypostprocfunc = func
      
  def run_pypostproc(self):
    '''
    Runs the Python post processing function.
    
    :rtype: data returned by pypostprocfunc
    '''
    import time
    from abapy.misc import load
    t0 = time.time()
    abqdata = load(self.workdir + self.simname + '.pckl')
    data = self.pypostprocfunc(abqdata)
    t1 = time.time()
    print '< Post Processed {0} in Python: duration {1:.2f}s>'.format(self.simname, t1 - t0)  
    return data 
    
class ContactData:
  '''
  ContactData class aims to store and proceed all contact related data:
  
  * Position of nodes involved in the contact relationship.
  * Contact pressure on these nodes.
  
  This class can be used to perform various tasks:
  
  * Find contact shape.
  * Compute contact area.
  * Find contact contour.
  * Produce matrix (AFM-like) images using the SPYM module.
  
:param repeat: this parameter is only used in the case of 3D contact. Due to symmetries, only a portion of the problem is computed. To get back to complete problem, a symmetry has to be performed and then a given number of copies of the result, this number is repeat. For example, to simulate a Vickers 4 sided indenter indentation, you will compute only 1 / 8 of the problem. So after a symmetry, you will need 4 copies of the result, the repeat = 4. To summarize, repeat is the number of faces of the indenter.
:type repeat: int > 0
:param is_3D: True for 3D contact, False for axisymmetric contact.
:type is_3D: bool
:param dtf: ``array.array`` data type for floats, 'f' for 32 bit float, 'd' for 64 float.
:type dtf: string
  
.. plot:: example_code/indentation/ContactData.py
   :include-source:    
  '''
  def __init__(self, repeat = 3, is_3D = False, dtf = 'f'):
    from array import array
    self.coor1 = array(dtf, [])
    self.coor2 = array(dtf, [])
    self.altitude = array(dtf, [])
    self.pressure = array(dtf, [])
    self.repeat = repeat
    self.is_3D = is_3D
    self.dtf = dtf
    
  def add_data(self,coor1, coor2 = 0., altitude = 0., pressure = 0.):
    '''
    Adds data to a ContactData instance.
    
    :param coor1: radial position in the axisymmetric case or in plane position first coordinate in the 3D case.
    :type coor1: float or list like
    :param coor2: orthoradial position in the axisymmetric case of second in plane coordinate in the 3D case.
    :type coor2: float of list like
    :param altitude: out of plane position.
    :type altitude: float or list like
    :param pressure: normal contact pressure.
    :type pressure: float or list like
    
    
    
    '''
    from array import array
    if type(coor1) in [float, int, long]:
      self.coor1.append(float(coor1))
      self.coor2.append(float(coor2))
      self.altitude.append(float(altitude))
      self.pressure.append(float(pressure))
    if hasattr(coor1, '__contains__'):
      coor1 = array(self.dtf, coor1)
      if hasattr(coor2, '__contains__'):
        coor2 = array(self.dtf, coor2)
      else: 
        coor2 = array(self.dtf, [coor2 for i in xrange(len(coor1))])
      if hasattr(altitude, '__contains__'):
        altitude = array(self.dtf, altitude)
      else: 
        altitude = array(self.dtf, [altitude for i in xrange(len(coor1))])
      if hasattr(pressure, '__contains__'):
        pressure = array(self.dtf, pressure)
      else: 
        pressure = array(self.dtf, [pressure for i in xrange(len(coor1))])
      self.coor1 += coor1
      self.coor2 += coor2
      self.altitude += altitude
      self.pressure += pressure
      
      
      
  def __repr__(self):
    pattern = '<ContactData instance w. {0} data points>'
    return pattern.format(len(self.coor1))
   
   
  def get_3D_data(self, axi_repeat = 100, delaunay_disp = None, crit_dist = 1.e-5):
    '''
    Returns full 3D data usable for interpolation or for ploting. This method performs all the copy and paste needed to get the complete contact (due to symmetries) and also producec a triangular mesh of the surface using the Delaunay algorithm (via ``scipy``).
    
:param axi_repeat: number of times axisymmetric profile has to be pasted orthoradially.
:type axi_repeat: int > 0
:rtype: 3 arrays points, alt, press and conn
    
.. plot:: example_code/indentation/ContactData-get_3D_data.py
  :include-source:    
    '''
    from scipy.spatial import Delaunay
    from numpy import sin, cos, pi, array, append, delete
    from copy import copy
    from mesh import get_neighbors
    c1, c2, alt, press = array([]),array([]),array([]),array([])
    if self.is_3D == False:
      delta = 2 *pi / float(axi_repeat)
      for i in xrange(axi_repeat):
        c1 = append( c1, cos(delta * i) * ( array(self.coor1) ))
        c2 = append( c2, sin(delta * i) * ( array(self.coor1) ))
        alt = append(alt, self.altitude)
        press = append(press, self.pressure)
    else:
      c1_0 = array(self.coor1)
      c2_0 = array(self.coor2)
      alt_0 = array(self.altitude)
      press_0 = array(self.pressure)
      # Applying symmetry
      c1_0 = append(c1_0, c1_0)
      c2_0 = append(c2_0, -c2_0)
      alt_0 = append(alt_0, alt_0)
      press_0 = append(press_0, press_0) 
      delta = 2 * pi / self.repeat
      for i in xrange(self.repeat):
        c1 = append(c1, cos(delta * i) * c1_0 - sin(delta * i ) * c2_0)
        c2 = append(c2, cos(delta * i) * c2_0 + sin(delta * i ) * c1_0)
        alt = append(alt, alt_0)
        press = append(press, press_0)
    
    
    points = array([c1, c2]).transpose()    
    neighbors = get_neighbors(points, crit_dist = crit_dist)
    to_delete = [ ]
    for neighbor in neighbors: to_delete += neighbor
    #print to_delete
    points = delete(points, to_delete, 0)  
    alt = delete(alt, to_delete, 0)  
    press = delete(press, to_delete, 0) 
    c1, c2 = points[:,0], points[:,1]
    if delaunay_disp != None:
      c1_temp, c2_temp = copy(c1), copy(c2)
      c1, c2 = delaunay_disp(c1, c2)
        
    points = array([c1, c2]).transpose()        
    conn = Delaunay(points).vertices
    if delaunay_disp != None: 
      c1, c2 = c1_temp, c2_temp
      points = array([c1, c2]).transpose()  
    return points, alt, press, conn
  
  def export2spym(self, lx, ly, xc = 0., yc = 0., nx = 256, ny = 256, xy_unit = 'm', alt_unit = 'm', press_unit = 'Pa', axi_repeat = 100, delaunay_disp = None, method = 'linear'):
    '''
    Exports data to ``spym.generic.Spm_image`` format.
    
    :param lx: length on x axis
    :type lx: float
    :param ly: length on y axis
    :type ly: float
    :param xc: position of the center of the image on the x axis
    :type xc: float
    :param yc: position of the center of the image on the y axis
    :type yc: float
    :param nx: x resolution
    :type nx: uint
    :param ny: y resolution
    :type ny: uint
    :param xy_unit: xy unit
    :type xy_unit: str
    :param alt_unit: altitude unit
    :type alt_unit: str
    :param press_unit: contact pressure unit
    :type press_unit: str
     
    See get_3D_data for other params.
    
    .. plot:: example_code/indentation/ContactData-export2spym.py
      :include-source:  
    
    This script also produces a GSF image file, readable by both ``spym`` and Gwyddion: :download:`image.gsf <example_code/indentation/image.gsf>`
    
    
    '''
    from spym.generic import Spm_image
    import numpy as np
    x = np.linspace(-lx/2., lx/2., nx) + xc 
    y = np.linspace(-ly/2., ly/2., ny) + yc 
    X, Y = np.meshgrid(x,y)
    alt, press = self.interpolate(X, Y, axi_repeat, delaunay_disp, method)
    Alt = Spm_image(
      name = 'Altitude', 
      data = alt, 
      lx = lx, ly = ly, 
      xc =lx/2., yc = ly/2., 
      xy_unit = xy_unit, 
      z_unit = alt_unit, 
      channel = 'Altitude')
    Press = Spm_image(
      name = 'Contact Pressure', 
      data = press, 
      lx = lx, ly = ly, 
      xc =lx/2., yc = ly/2., 
      xy_unit = xy_unit, 
      z_unit = press_unit, 
      channel = 'Pressure')
    return Alt, Press
      
  def interpolate(self, coor1, coor2, axi_repeat = 100, delaunay_disp = None, method = 'linear'):
    '''
    Allows general interpolation on the a contact data instance. 
    
  :param coor1: radial position in the axisymmetric case or in plane position first coordinate in the 3D case.
  :type coor1: any list/array of floats
  :param coor2: orthoradial position in the axisymmetric case of second in plane coordinate in the 3D case.
  :type coor2: any list/array of floats
  :param axi_repeat: number of times axisymmetric profile has to be pasted orthoradially.
  :type axi_repeat: int > 0
    
  .. plot:: example_code/indentation/ContactData-interpolate.py
    :include-source:   
    '''
    
    from scipy.interpolate import griddata
    import numpy as np
    points, alt, press, conn = self.get_3D_data(axi_repeat = axi_repeat, delaunay_disp = delaunay_disp)
    Alt = griddata(points, alt, (coor1, coor2), method=method)
    Press = griddata(points, press, (coor1, coor2), method=method)
    return Alt, Press
     
  def contact_area(self, delaunay_disp = None):
    '''
    Returns the cross area of contact using the contact pressure field. The contact area is computed using a Delaunay triangulation of the available data points. This triangulation can be oriented using the delaunay_disp option (use very carefuly).     
    '''
    
    import numpy as np
    if self.is_3D: 
      # Computing Triangle Area
      def triangle_area(points,conn):
        x, y = points[:,0], points[:,1]
        area = 0. * np.arange(len(conn))
        for i in xrange(len(conn)):
          tri = conn[i]
          xt = x[tri]
          yt = y[tri]
          xb, xc = xt[1] - xt[0],  xt[2] - xt[0]
          yb, yc = yt[1] - yt[0],  yt[2] - yt[0]
          area[i] = .5 * abs( xb * yc - xc * yb )
        return area    
      
      # Computing numerical 2D integration
      def triangle_integration(conn, areas, field, coeffs = [0., 1./3., 2./3., 1.]):
        out = 0.
        for i in xrange(len(conn)):
          tri = conn[i]
          area = areas[i]
          f = field[tri].sum()
          out += coeffs[f] * area 
        return out
      
      # Computing the area of all triangle where all summits have a non zero pressure
      def triangle_non_zero(conn, areas, field):
        out = 0.
        for i in xrange(len(conn)):
          tri = conn[i]
          area = areas[i]
          f = field[tri]
          out += int(f.sum() / 3.) * area
        return out
      
      points, alt, press, conn = self.get_3D_data(delaunay_disp = delaunay_disp)
      areas = triangle_area(points,conn)
      norm_press = (press > 0.) + 0
      #out1 = triangle_integration(conn, areas, norm_press, coeffs = [0., 0., 0., 1.])
      out1 = triangle_integration(conn, areas, norm_press, coeffs = [0., 1./3., 2./3., 1.])
      #out2 = triangle_integration(conn, areas, norm_press, coeffs = [0., 1., 1., 1.])
      #out = triangle_non_zero(conn, areas, norm_press)
      #return np.array([out0, out1, out2])
      return out1
    else:
      r = np.array(self.coor1)
      p = np.array(self.pressure)
      loc = r.argsort()
      r = r[loc]
      p = p[loc]
      loc = np.where(p !=0.)[0].max()
      rc = r[loc]
      Ac = np.pi * rc**2  
      return Ac
    
  def contact_contour(self, delaunay_disp = None):
    '''
    Returns the contour of the contact zone.
    '''
    
    from matplotlib import pyplot as plt
    points, alt, press, conn = self.get_3D_data(delaunay_disp = delaunay_disp)
    norm_press = (press > 0.) + 0
    x, y = points[:,0], points[:,1]
    fig = plt.figure()
    c = plt.tricontour(x, y, conn, norm_press, levels = [0.0001,0.5, 0.9999 ]).collections
    points0 = c[0].get_paths()[0].vertices
    points1 = c[1].get_paths()[0].vertices
    points2 = c[2].get_paths()[0].vertices
    plt.close()
    return points2, points1, points0
  
  def max_altitude(self):
    '''
    Returns the maximum altitude.
    '''
    return max(self.altitude)
  
  def min_altitude(self):
    '''
    Returns the minimum altitude.
    '''
    return min(self.altitude)
    
  def max_pressure(self):
    '''
    Returns the maximum pressure.
    '''
    return max(self.pressure)
    
  def min_pressure(self):
    '''
    Returns the minimum pressure.
    '''
    return min(self.pressure)
  
  def contact_radius(self, zero_pressure = 0.):
    """
    Returns the contact radius in 2D cases, nan otherwise.
    """
    import numpy as np
    if self.is_3D: 
      return np.nan
    else:
      r = np.array(self.coor1)
      z = np.array(self.altitude)
      p = np.array(self.pressure)
      loc = np.where(p>zero_pressure)[0]
      if len(loc) != 0:
        rc = r[loc].max()
        return rc
      else:
        return np.nan
      
  def contact_height(self, zero_pressure = 0.):
    """
    Returns the contact height in 2D cases, nan otherwise.
    """
    import numpy as np
    if self.is_3D: 
      return np.nan
    else:
      r = np.array(self.coor1)
      z = np.array(self.altitude)
      p = np.array(self.pressure)
      loc = np.where(p>zero_pressure)[0]
      if len(loc) != 0:
        hc = z[loc].max()
        return hc
      else:
        return np.nan       

    

def Get_ContactData(odb, instance, node_set):
  '''
  Finds and reformulate contact data on a given node set and a give instance. This function aims to read in Abaqus odb files and is then only usable in ``abaqus python`` and ``abaqus viewer -noGUI``. Following conventions are used:
  
  * The normal to the initial surface must be the y axis. 
  * In axisymmetrical simulations, this is nearly automatic. In 3D simulations, the initial plane surface must be in parallel to the (x,z) plane. 
  * As a consequence, coor1 will be x, coor2 will be z and the altitude is y.
  
  :param odb: the odb instance where needed data is.
  :type odb: odb instance obtained using ``odbAccess.openOdb``.
  :param instance: name of an instance in contact.
  :type instance: string
  :param node_set: name of a node set belonging to the instance.
  :type node_set: string
  '''
  from abapy.postproc import GetHistoryOutputByKey as gho
  from abapy.indentation import ContactData
  from array import array
  from abaqusConstants import THREE_D
  # Checking embedded space
  is_3D = False # Let's suppose that the instance is axisymmetric
  if odb.rootAssembly.instances[instance].embeddedSpace == THREE_D: is_3D = True
  # Finding nodes in the specified node set
  NodeSet = odb.rootAssembly.instances[instance].nodeSets[node_set].nodes
  topNodes = [node.label for node in NodeSet] # Labels of the contact nodes
  # let's check which nodes have available contact history
  hist_region_key_pattern = 'Node ' + instance + '.' 
  for label in topNodes:
    available_hist = odb.steps[odb.steps.keys()[0]].historyRegions[hist_region_key_pattern + str(label)].historyOutputs.keys()
    cpress_available = False
    for key in available_hist:
      if 'CPRESS' in key: 
        cpress_available = True
        hist_target_key = key
    if cpress_available == False: topNodes.remove(label) 
  # Getting data
  coor1 = gho(odb, 'COOR1')
  if is_3D: coor2 = gho(odb, 'COOR3') # Only in 3D
  altitude = gho(odb, 'COOR2')
  pressure = gho(odb, hist_target_key)
  # Reformating data
  keys = coor1.keys()
  Coor1, Coor2, Altitude, Pressure = [], [], [], []
  for k in keys:
    Coor1.append(coor1[k].data)
    if is_3D: Coor2.append(coor2[k].data)
    Altitude.append(altitude[k].data)
    Pressure.append(pressure[k].data)
  n_nodes = len(keys) # Number of contact nodes
  n_steps = len(Coor1[0]) # Number of available steps
  out = []
  for st in xrange(n_steps):
    out.append([])
    out_step = out[-1]
    for fr in xrange(len(Coor1[0][st])): # fr in the indice of the frame
      out_step.append(ContactData(is_3D = is_3D))
      cd = out_step[-1]
      coor1_temp = array('d',[])
      coor2_temp = array('d',[])
      altitude_temp = array('d',[])
      pressure_temp = array('d',[])
      for n in xrange(n_nodes):
        coor1_temp.append(Coor1[n][st][fr])
        if is_3D: 
          coor2_temp.append(Coor2[n][st][fr])
        else:
          coor2_temp.append(0.)
        altitude_temp.append(Altitude[n][st][fr])
        pressure_temp.append(Pressure[n][st][fr])
      cd.add_data(coor1 = coor1_temp, coor2 = coor2_temp, altitude = altitude_temp, pressure = pressure_temp)
  
  return out
  
  
def equivalent_half_angle(half_angle, sweep_angle):
  '''
  Returns the half angle (in degrees) of the equivalent cone in terms of cross area and volume of a pyramidal indenter.
  :param half_angle: indenter half angle in degrees
  :type half_angle: float
  :param sweep_angle: sweep angle in degrees
  :type sweep_angle: float
  :rtype: float
  '''
  import numpy as np
  psi = np.radians(half_angle)
  alpha = np.radians(sweep_angle)
  Ap = 2 * np.pi / alpha * np.tan(psi)**2 *np.tan(alpha) / 2.
  rc = ( Ap/np.pi )**.5
  phi = np.arctan(rc)
  return np.degrees(phi)
  
# Hertz contact
# Author: A. Faivre

class Hertz(object):
  '''
  Hertz spherical indentation model.
 
  .. plot:: example_code/indentation/Hertz.py
     :include-source:   
  '''
  def __init__(self, F = 1., a = None, h = None, R = 1., E = 1., nu = 0.3):
    self.R = R
    self.E = E
    self.nu = nu
    
    
    if (a, F, h) == (None, None, None):
      raise ValueError('a, F or h are not define, one at least must be defined')

    if a != None:
      self.a = a
      self.F = self.set_force()
      
    if F != None and a == None:
      self.F = F
      self.a = self.set_contact_r()
      
    if h != None and a == None and F == None:
      self.h = h
      self.a = self.set_depth_a()    
      self.F = self.set_depth_F()
    
# with a         
  def set_force(self):
    F = (4. * self.Eeq * self.a**3) / (3. * self.R)
    return F

# with F    
  def set_contact_r(self):
    a = ( (3.*self.R*self.F)/(4.*self.Eeq) )**(1./3.)      
    return a

# with h 
  def set_depth_a(self):
    a = ( self.R**2 - (self.R - self.h)**2 )**.5
    return a
      
  def set_depth_F(self):
    F = (4. * self.Eeq * self.a**3) / (3. * self.R)
    return F
  
# Some outputs   
  def get_Eeq(self):
    '''Eeq: equivalent modulus (GPa)'''
    Eeq = self.E/(1.-self.nu**2)
    return Eeq
  Eeq = property(get_Eeq)
    
  def get_mean_p(self): 
    from math import pi   
    '''pm: mean pressure (GPa)'''
    pm = self.F / (pi*self.a**2)
    return pm
  mean_p = property(get_mean_p)
    
#Sigma
  def sigma(self, r, z, t = 0., labels = None):
    '''
    To do.
    '''
    from numpy import zeros_like
    return self.sigma_rr(r,z,t), self.sigma_zz(r,z,t), self.sigma_tt(r,z,t), self.sigma_rz(r,z,t), 0. * r, 0. * r 
    
#Sigma rr
  def sigma_rr(self, r, z, t = 0., labels = None):
    from numpy import array, float64, arctan
    a = self.a
    v = self.nu
    pm = self.mean_p
    r, z = array(r), array(z)
    #r, z = remove_zeros(r), remove_zeros(z) 
    b = r**2 + z**2 - a**2
    u = 1./2. * ( b + ( b**2 + (2.*a*z)**2 )**.5 )           
    return 3./2.*( (1.-2.*v)/3. * (a/r)**2 * (1.- (z/(u**.5))**3) + (z/(u**.5))**3 * ( (a**2*u)/(u**2+(a*z)**2) ) + z/(u**.5) * (u * (1.-v)/(a**2 +u) + (1.+v) * (u**.5)/a * arctan(a/(u**.5)) - 2.) ) * pm
    
    
    return set_sigma_rr

#Sigma zz
  def sigma_zz(self, r, z, t = 0., labels = None):
    from numpy import array, float64
    a = self.a
    v = self.nu
    pm = self.mean_p
    r, z = array(r), array(z)
    #r, z = remove_zeros(r), remove_zeros(z) 
    b = (r**2 + z**2 - a**2)
    u = 1./2. * ( b + ( b**2 + (2.*a*z)**2 )**.5 ) 
    return -3./2. * (z/(u**.5))**3 * ( (u * a**2)/(u**2 + (a*z)**2) ) * pm
    

#Sigma tt
  def sigma_tt(self, r, z, t = 0., labels = None):
    from numpy import array, float64, arctan
    a = self.a
    v = self.nu
    pm = self.mean_p
    r, z = array(r), array(z)
    #r, z = remove_zeros(r), remove_zeros(z) 
    b = (r**2 + z**2 - a**2)
    u = 1./2. * ( b + ( b**2 + (2.*a*z)**2 )**.5 ) 
    return -3./2. * ( (1.-2.*v)/3. * (a/r)**2 * (1.-(z/u**.5)**3) + z/u**.5 * (2.*v + u*(1.-v)/(a**2+u) - (1.+v)*(u**.5/a)*arctan(a/u**.5))) * pm
       

#Sigma_rz
  def sigma_rz(self, r, z, t = 0., labels = None):
    from numpy import array, float64
    a = self.a
    v = self.nu
    pm = self.mean_p
    r, z = array(r), array(z)
    #r, z = remove_zeros(r), remove_zeros(z) 
    b = (r**2 + z**2 - a**2)
    u = 1./2. * ( b + ( b**2 + (2.*a*z)**2 )**.5 ) 
    return -3./2. * ( (r * z**2) / (u**2 + (a*z)**2) ) * ( (a**2 * u**.5) / ( a**2 + u) ) * pm
    
  
'''
HANSON conical indentation
===============
'''


class Hanson(object):
  '''
  Hanson conical indentation model.
 
  .. plot:: example_code/indentation/Hanson.py
     :include-source:   
  '''
  def __init__(self, F = None, a = None, h = None, half_angle = 70.29, E = 1., nu = 0.3):
    self.E = E
    self.nu = nu
    self.half_angle = half_angle
   
 
    if (a, F, h) == (None, None, None):
      raise ValueError('a, F or h are not defined, at least one must be defined')
    if a != None:
      self.a = a
    if F != None:
      self.F = F
    if h != None:
      self.h = h    
    


    

# Managing Force    
  def set_F(self, F):
    from math import tan, radians
    self.a  = ( 2. * self.H * F *tan(radians(self.half_angle)) )**.5  
  def get_F(self):
    from math import tan, radians
    F = self.a**2 / ( 2.*self.H * tan(radians(self.half_angle)))
    return F  
  F = property(get_F, set_F)  


    
# Managing penetration
  def set_h(self, h):
    from math import pi, tan, radians
    self.a = 2. / pi * tan(radians(self.half_angle)) * h
  def get_h(self):
    return pi * self.a / ( tan(radians(self.half_angle)) * 2.)    
  h = property(get_h, set_h)
  
# Model internal parameters   
  def get_H(self):
    from math import pi
    H = (1.-self.nu**2) / (pi * self.E)
    return H
  H = property(get_H)
  
  def get_epsilon(self):
    from math import tan, radians
    return self.a / tan(radians(self.half_angle))
  epsilon = property(get_epsilon)
  
  def get_Eeq(self):
    '''Eeq: equivalent modulus (GPa)'''
    Eeq = self.E/(1.-self.nu**2)
    return Eeq
  Eeq = property(get_Eeq)  
  
  def  l1(self, r, z, t =0.):
    a = self.a  
    return 1./2. * ( ((r+a)**2 + z**2)**.5 - ((r-a)**2 + z**2)**.5 )
  
  
  def l2(self, r, z, t =0.):
    a = self.a
    return 1./2. * ( ((r+a)**2 + z**2)**.5 + ((r-a)**2 + z**2)**.5 )
  

#Sigma
  def sigma(self, r, z, t = 0., labels = None):
    '''
    To do.
    '''
    from numpy import zeros_like
    return self.sigma_rr(r,z,t), self.sigma_zz(r,z,t), self.sigma_tt(r,z,t), self.sigma_rz(r,z,t), 0. * r, 0. * r
    
#Sigma 1
  def sigma_1(self, r, z, t = 0., labels = None):
    from numpy import array, log
    E = self.E
    e = self.epsilon
    a = self.a
    v = self.nu
    r, z = array(r), array(z)
    l1, l2 = self.l1(r,z),  self.l2(r,z)
    return - (E*e)/(2.*a*(1.-v**2))*( (1.+2.*v)*log((l2+(l2**2-r**2)**.5)/(z+(r**2+z**2)**.5))+ z*((l2**2-a**2)**.5/(l2**2-l1**2)-1./(r**2+z**2)**.5))
    
  
    
#Sigma 2
  def sigma_2(self, r, z, t = 0., labels = None):
    from numpy import array, log
    E = self.E
    e = self.epsilon
    a = self.a
    v = self.nu
    r, z = array(r), array(z)
    l1, l2 = self.l1(r,z),  self.l2(r,z)
    return - (E*e)/(2.*a*(1.-v**2))*( (1.-2.*v)*((1./(a*(r**2)))*(2.*a**2-l2**2)*(a**2-l1**2)**.5 + (z*(r**2+z**2)**.5)/r**2 - a**2/r**2) + (z**2*a*(r**2+2.*l1**2-2.*l2**2))/(r**2*(a**2-l1**2)**.5*(l2**2-l1**2)) + (z*(r**2+2.*z**2))/(r**2*(r**2+z**2)**.5) )     
      
    
#Sigma zz   
  def sigma_zz(self, r, z, t = 0., labels = None):
    from numpy import array, log
    E = self.E
    e = self.epsilon
    a = self.a
    v = self.nu
    r, z = array(r), array(z)
    l1, l2 = self.l1(r,z),  self.l2(r,z)
    return - (E*e)/(2.*a*(1.-v**2))*( log((l2+(l2**2-r**2)**.5) / (z+(r**2+z**2)**.5)) - (z**2*l2)/((l2**2-r**2)**.5*(l2**2-l1**2)) + z/(r**2+z**2)**.5)
   
    
#Sigma rz
  def sigma_rz(self, r, z, t = 0., labels = None):
    from numpy import array, log
    E = self.E
    e = self.epsilon
    a = self.a
    v = self.nu
    r, z = array(r), array(z)
    l1, l2 = self.l1(r,z),  self.l2(r,z)
    return - (E*e)/(2.*a*(1.-v**2)) * 1./r * ( (z*l2*(l2**2-r**2)**.5/(l2**2-l1**2)) - z**2/(r**2+z**2)**.5)
    
    
#Sigma xx
  def sigma_rr(self, r, z, t = 0., labels = None):
    return .5 * ( self.sigma_1(r, z) + self.sigma_2(r, z) )

#Sigma tt
  def sigma_tt(self, r, z, t = 0., labels = None):
    return .5 * ( self.sigma_1(r, z) - self.sigma_2(r, z) )    
    

        
