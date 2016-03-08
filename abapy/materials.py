'''
Materials
=========
'''

def float_arg(arg):
  """
  Tests if an arg is float or convertible to float
  """
  try:
    arg = [float(arg)]
  except:
    pass
  return arg    
  

class VonMises(object):
  ''' 
  Represents von Mises materials used for FEM simulations
  
  :param E: Young's modulus.
  :type E: float, list, array.array
  :param nu: Poisson's ratio.
  :type nu: float, list, array.array
  :param sy: Yield stress.
  :type sy: float, list, array.array
  
  .. note:: 
     All inputs must have the same length or an exception will be raised.
     
     
  >>> from abapy.materials import VonMises
  >>> m = VonMises(labels='myMaterial',E=1,nu=0.45, sy=0.01)
  >>> print m.dump2inp()
  ...
  
  
  '''
  def __init__(self, labels='mat', E = 1., nu = 0.3, sy = 0.01, dtf='d'):
    from array import array
    import numpy
    if type(labels) is str: labels=[labels]
    self.labels=labels
    l = len(labels)
    E = float_arg(E)
    if len(E) != l: raise Exception, 'Parameters must all have the same length'
    self.E=array(dtf,E)
    nu = float_arg(nu)
    if len(nu) != l: raise Exception, 'Parameters must all have the same length'
    self.nu=array(dtf,nu)  
    sy = float_arg(sy)
    if len(sy) != l: raise Exception, 'Parameters must all have the same length'
    self.sy=array(dtf,sy)
  def __repr__(self):
    return '<VonMises instance: {0} samples>'.format(len(self.E))
  def dump2inp(self):
    '''
    Returns materials in INP format suitable with abaqus input files.
    
    :rtype: string
    '''
    out = '** {0}\n'.format(self.__repr__())
    pattern = '*MATERIAL, NAME={0}\n*ELASTIC\n  {1}, {2}\n*PLASTIC\n  {3}, 0.\n'
    for i in xrange(len(self.E)):
      out += pattern.format(self.labels[i],self.E[i],self.nu[i],self.sy[i])
    return out[0:-1]

class Elastic(object):
  ''' 
  Represents an istotrop linear elastic material used for FEM simulations
  
  :param E: Young's modulus.
  :type E: float, list, array.array
  :param nu: Poisson's ratio.
  :type nu: float, list, array.array
  
  .. note:: 
     All inputs must have the same length or an exception will be raised.
  '''
  def __init__(self, labels='mat', E = 1., nu = 0.3, dtf='d'):
    from array import array
    if type(labels) is str: labels=[labels]
    self.labels=labels
    l = len(labels)
    E = float_arg(E)
    if len(E) != l: raise Exception, 'Parameters must all have the same length'
    self.E=array(dtf,E)
    nu = float_arg(nu)
    if len(nu) != l: raise Exception, 'Parameters must all have the same length'
    self.nu=array(dtf,nu)  
  def __repr__(self):
    return '<Elastic instance: {0} samples>'.format(len(self.E))
  def dump2inp(self):
    '''
    Returns materials in INP format suitable with abaqus input files.
    
    :rtype: string
    '''
    out = '** {0}\n'.format(self.__repr__())
    pattern = '*MATERIAL, NAME={0}\n*ELASTIC\n  {1}, {2}\n'
    for i in xrange(len(self.E)):
      out += pattern.format(self.labels[i],self.E[i],self.nu[i])
    return out[0:-1]


class DruckerPrager(object):
  ''' 
  Represents Drucker-Prager materials used for FEM simulations
  
  :param E: Young's modulus.
  :type E: float, list, array.array
  :param nu: Poisson's ratio.
  :type nu: float, list, array.array
  :param sy: Compressive yield stress.
  :type sy: float, list, array.array
  :param beta: Friction angle in degrees.
  :type beta: float, list, array.array
  :param psi: Dilatation angle in degress. If psi = beta, the plastic flow is associated. If psi = None, the associated flow is automatically be chosen.
  :type psi: float, list, array.array or None
  :param k: tension vs. compression asymmetry. For k = 1., not asymmetry, for k=0.778 maximum possible asymmetry.
  :type k: float, list, array.array
  
  .. note:: 
     All inputs must have the same length or an exception will be raised.
     
   ...
  
  
  '''
  def __init__(self, labels='mat', E = 1., nu = 0.3, sy = 0.01, beta = 10., psi = None, k = 1., dtf='d'):
    from array import array
    if type(labels) is str: labels=[labels]
    self.labels=labels
    l = len(labels)
    E = float_arg(E)
    if len(E) != l: raise Exception, 'Parameters must all have the same length'
    self.E=array(dtf,E)
    nu = float_arg(nu)
    if len(nu) != l: raise Exception, 'Parameters must all have the same length'
    self.nu=array(dtf,nu)  
    sy = float_arg(sy)
    if len(sy) != l: raise Exception, 'Parameters must all have the same length'
    self.sy=array(dtf,sy)
    beta = float_arg(beta)
    if len(beta) != l: raise Exception, 'Parameters must all have the same length'
    self.beta = array(dtf,beta)
    if psi == None: psi = beta
    psi = float_arg(psi)
    if len(psi) != l: raise Exception, 'Parameters must all have the same length'
    self.psi = array(dtf,psi)
    k = float_arg(k)
    if len(k) != l: raise Exception, 'Parameters must all have the same length'
    self.k = array(dtf,k)
  
  def __repr__(self):
    return '<DruckerPrager instance: {0} samples>'.format(len(self.E))
  
  def dump2inp(self):
    '''
    Returns materials in INP format suitable with abaqus input files.
    
    :rtype: string
    '''
    out = '** {0}\n'.format(self.__repr__())
    pattern = '*MATERIAL, NAME={0}\n*ELASTIC\n  {1}, {2}\n*DRUCKER PRAGER\n  {3}, {4}, {5}\n*DRUCKER PRAGER HARDENING\n  {6}, 0.\n'
    for i in xrange(len(self.E)):
      out += pattern.format(
        self.labels[i],
        self.E[i],
        self.nu[i],
        self.beta[i],
        self.k[i],
        self.psi[i],
        self.sy[i])
    return out[0:-1]
    
class Hollomon(object):
  r''' 
  Represents von Hollom materials (i. e. power law haderning and von mises yield criterion) used for FEM simulations.
  
  :param E: Young's modulus.
  :type E: float, list, array.array
  :param nu: Poisson's ratio.
  :type nu: float, list, array.array
  :param sy: Yield stress.
  :type sy: float, list, array.array
  :param n: hardening exponent
  :type sy: float, list, array.array
  :param kind: kind of equation to be used (see below). Default is 1.
  :type kind: int  
  
  .. note:: 
     All inputs must have the same length or an exception will be raised.
  
  Several sets of equations are refered to as Hollomon stress-strain law. In all cases, we the strain decomposition :math:`\epsilon = \epsilon_e + \epsilon_p`  is used and the elastic part is described by :math:`\sigma = E \epsilon_e = E \epsilon`. Only the plastic parts (i. e. :math:`\sigma > \sigma_y`) differ:
  
  * kind 1: 
  
  .. math::
  
     \sigma = \sigma_y \left( \epsilon E / \sigma_y \right)^n
  
  * kind 2:
    
  .. math::
  
     \sigma = \sigma_y \left( 1 + \epsilon_p \right)^n = E \epsilon_e 
       
  .. plot:: example_code/materials/Hollomon.py
    :include-source:   
  '''
  def __init__(self, labels='mat', E = 1., nu = 0.3, sy = 0.01, n = 0.2, kind = 1, dtf='d'):
    from array import array
    import numpy
    if type(labels) is str: labels=[labels]
    self.labels=labels
    l = len(labels)
    E = float_arg(E)
    if len(E) != l: raise Exception, 'Parameters must all have the same length'
    self.E=array(dtf,E)
    nu = float_arg(nu)
    if len(nu) != l: raise Exception, 'Parameters must all have the same length'
    self.nu=array(dtf,nu)  
    sy = float_arg(sy)
    if len(sy) != l: raise Exception, 'Parameters must all have the same length'
    self.sy=array(dtf,sy)
    n = float_arg(n)
    if len(n) != l: raise Exception, 'Parameters must all have the same length'
    self.n=array(dtf,n)
    self.kind = kind
  def __repr__(self):
    return '<Hollomon instance: {0} samples>'.format(len(self.E))
  
  def get_table(self, position=0, eps_max = 10., N = 100):
    '''
    Returns the tabular data corresponding to the tensile stress strain law using log spacing.
    :param position: indice of the concerned material (default is 0).
    :type position: int
    :param eps_max: maximum strain to be computed. If kind is 1, eps_max is the total strain, if kind is 2, eps_max is the plastic strain.
    :type eps_max: float
    :param N: number of points to be computed.
    :type N: int
    :rtype: ``numpy.array``
    '''
    import numpy as np
    sy = self.sy[position]
    E = self.E[position]
    n = self.n[position]
    if self.kind == 1:
      ey = sy/E
      s = 10.**np.linspace(0., np.log10(eps_max/ey), N, endpoint = True)
      eps = ey * s
      sigma = sy * s**n
    if self.kind == 2:
      #eps_p = np.logspace(0., np.log10(eps_max +1.), N) -1.
      s = np.linspace(0., eps_max**n, N)
      eps_p = s**(1/n)
      sigma = sy * (1. + s)
      eps = eps_p + sigma / E
    return np.array([eps, sigma]).transpose()
      
  def dump2inp(self, eps_max = 10., N = 100):
    '''
    Returns materials in INP format suitable with abaqus input files.
    
    :param eps_max: maximum strain to be computed.
    :type eps_max: float
    :param N: number of points to be computed.
    :type N: int
    :rtype: string
    '''
    out = '** {0}\n'.format(self.__repr__())
    pattern = '*MATERIAL, NAME={0}\n*ELASTIC\n  {1}, {2}\n*PLASTIC\n{3}\n'
    for i in xrange(len(self.E)):
      table = self.get_table(position = i, eps_max = eps_max, N = N)
      sigma = table[:,1]
      eps = table[:,0]
      #eps_p = eps - eps[0]
      eps_p = [eps[j] - sigma[j] / self.E[i] for j in xrange(len(eps))]
      data = ''
      for j in xrange(len(table)):
        data += '  {0}, {1}\n'.format(sigma[j], eps_p[j])
      out += pattern.format(self.labels[i],self.E[i],self.nu[i],data[0:-1])
    return out[0:-1]
    
    
    
class Bilinear(object):
  ''' 
  Represents von Mises materials used for FEM simulations
  
  :param E: Young's modulus.
  :type E: float, list, array.array
  :param nu: Poisson's ratio.
  :type nu: float, list, array.array
  :param Ssat: Saturation stress.
  :type Ssat: float, list, array.array
  :param n: Slope of the first linear plastic law
  :type n: float, list, array.array
  :param Sy: Stress at zero plastic strain
  :type Sy: float, list, array.array
  
  .. note:: 
     All inputs must have the same length or an exception will be raised.
     
  '''
  def __init__(self, labels='mat', E = 1., nu = 0.3, Ssat = 1000., n=100., sy=100. ,dtf='d'):
    from array import array
    import numpy
    
    if type(labels) is str: labels=[labels]
    self.labels=labels
    l = len(labels)
    E = float_arg(E)
    if len(E) != l: raise Exception, 'Parameters must all have the same length'
    self.E=array(dtf,E)
    nu = float_arg(nu)
    if len(nu) != l: raise Exception, 'Parameters must all have the same length'
    self.nu=array(dtf,nu)  
    Ssat = float_arg(Ssat)
    if len(Ssat) != l: raise Exception, 'Parameters must all have the same length'
    self.Ssat=array(dtf,Ssat)
    n = float_arg(n)
    if len(n) != l: raise Exception, 'Parameters must all have the same length'
    self.n=array(dtf,n)
    sy = float_arg(sy)
    if len(sy) != l: raise Exception, 'Parameters must all have the same length'
    self.sy=array(dtf,sy)    
    
  def __repr__(self):
    return '<Bilinear instance: {0} samples>'.format(len(self.E))
  
  def dump2inp(self):
    '''
    Returns materials in INP format suitable with abaqus input files.
    
    :rtype: string
    '''
    out = '** {0}\n'.format(self.__repr__())
    pattern = '*MATERIAL, NAME={0}\n*ELASTIC\n  {1}, {2}\n*PLASTIC\n  {3}, 0.\n {4}, {5} \n'
    Eps_p_sat=[]
    for i in xrange(len(self.E)):
        Eps_p_sat.append(abs(self.Ssat[i] - self.sy[i])/self.n[i])
        if self.sy[i] > self.Ssat[i]:
          self.sy[i] = self.Ssat[i]   
        out += pattern.format(self.labels[i],self.E[i],self.nu[i],self.sy[i], self.Ssat[i], Eps_p_sat[i])
    return out[0:-1]    

class SiDoLo(object):
  ''' 
  Represents constitutive equations via UMAT subroutines and SiDoLo used for FEM simulations
  
  :param: 
  :type : 
  
  .. note:: 
     All inputs must have the same length or an exception will be raised.
    
  '''
  def __init__(self, labels='mat', dictionary={}, umat={}, dirsid = '/tmp'):
    from array import array
    if type(labels) is str: labels=[labels]
    self.labels=labels
    self.dictionary=[dictionary]
    self.umat=[umat]
    self.dirsid=dirsid

  def __repr__(self):
    return '<UMAT/SiDoLo instance: {0} samples>'.format(len(self.labels))

  def dump2inp(self):
    '''
    Returns materials in a format suitable with Abaqus input files.
    
    :rtype: string
    '''  
    self.dump2coe()
    out = '** {0}\n'.format(self.__repr__()) # Initialisation
    pattern = '*MATERIAL, NAME={0}\n*USER MATERIAL, CONSTANTS=2\n  {1}, {2}\n*DEPVAR\n  {3}\n'
    for i in xrange(len(self.labels)):
      params  = self.dictionary[i]['Parameters']
      varint  = self.dictionary[i]['Variables']
      Lvarint = len(varint)
      out += pattern.format(self.labels[i],params['epsrk'],params['toly'],Lvarint)
    return out[0:-1] # La sortie est tout sauf le dernier element de out (ligne blanche)

  def dump2coe(self):
    '''
    Returns materials coefficients in a format suitable with Abaqus/SiDoLo.
    
    :rtype: string
    '''
    from collections import OrderedDict
    for i in xrange(len(self.labels)):
      # Definition des sous-dictionnaires
      coeffs  = self.dictionary[i]['Coefficients']
      varint  = self.dictionary[i]['Variables']

      # Ecriture dans un fichier formatte
      
      if self.dirsid[-1]!='/': self.dirsid = self.dirsid + '/'
      filename = self.dirsid +  self.labels[i] + '.coe'
      
      fich = open(filename, 'w')
      fich.write('*list \n')
      motif = ' {:s}'
      for k, v in coeffs.iteritems():
          fich.write(motif.format(k))
      fich.write('\n')
      fich.write('*values \n')
      motif = ' {0:s} {1:f}\n'
      for k, v in coeffs.iteritems():
          fich.write(motif.format(k,v))
      fich.close()
    return

class Ludwig(object):
  ''' 
  Represents Hollomon power law hardening used for FEM simulations.
  
  :param E: Young's modulus.
  :type E: float, list, array.array
  :param nu: Poisson's ratio.
  :type nu: float, list, array.array
  :param sy: Yield stress.
  :type sy: float, list, array.array
  :param n: hardening exponent
  :type n: float, list, array.array
  :param K: strenght index
  :type K: float, list, array.array
  
  .. note:: 
     All inputs must have the same length or an exception will be raised.
    
  .. plot:: example_code/materials/Hollomon.py
    :include-source:   
  
  
  '''
  def __init__(self, labels='mat', E = 1., nu = 0.3, sy = 150, K = 20., n = 0.2, dtf='d'):
    from array import array
    import numpy as np
    if type(labels) is str: labels=[labels]
    self.labels=labels
    l = len(labels)
    E = float_arg(E)
    if len(E) != l: raise Exception, 'Parameters must all have the same length'
    self.E=array(dtf,E)
    nu = float_arg(nu)
    if len(nu) != l: raise Exception, 'Parameters must all have the same length'
    self.nu=array(dtf,nu)  
    K = float_arg(K)
    if len(K) != l: raise Exception, 'Parameters must all have the same length'
    self.K=array(dtf,K)
    n = float_arg(n)
    if len(n) != l: raise Exception, 'Parameters must all have the same length'
    self.n=array(dtf,n)
    sy = float_arg(sy)
    if len(sy) != l: raise Exception, 'Parameters must all have the same length'
    self.sy=array(dtf,sy)
  def __repr__(self):
    return '<Hollomon instance: {0} samples>'.format(len(self.E))
  
  def get_table(self, position, eps_max = 1., N = 100):
    '''
    Returns the tabular data corresponding to the tensile stress strain law using log spacing.
    
    :param eps_max: maximum strain to be computed.
    :type eps_max: float
    :param N: number of points to be computed.
    :type N: int
    :rtype: ``numpy.array``
    '''
    import numpy as np
    K = self.K[position]
    n = self.n[position]
    sy = self.sy[position]
    E = self.E[position]
    ey = sy/E
    s = 10.**np.linspace(0., np.log10(eps_max/ey), N, endpoint = True)
    eps_p_temp, sigma = [0],[sy]
    
    for i in range(0, len(s)):
      eps_p_temp.append(ey * s[i])
      sigma.append(sy + K * (ey * s[i])**n)
      eps_p = np.asarray(eps_p_temp)
    return np.array([eps_p, sigma]).transpose()
      
  def dump2inp(self, eps_max = 1., N = 100):
    '''
    Returns materials in INP format suitable with abaqus input files.
    
    :param eps_max: maximum strain to be computed.
    :type eps_max: float
    :param N: number of points to be computed.
    :type N: int
    :rtype: string
    '''
    out = '** {0}\n'.format(self.__repr__())
    pattern = '*MATERIAL, NAME={0}\n*ELASTIC\n  {1}, {2}\n*PLASTIC\n{3}\n'
    for i in xrange(len(self.E)):
      table = self.get_table(position = i, eps_max = eps_max, N = N)
      sigma = table[:,1]
      eps_p = table[:,0]
      #eps = [eps_p[j] + sigma[j] / self.E[i] for j in xrange(len(eps_p))]
      data = ''
      for j in xrange(len(table)):
        data += '  {0}, {1}\n'.format(sigma[j], eps_p[j])
      out += pattern.format(self.labels[i],self.E[i],self.nu[i],data[0:-1])
    return out[0:-1]
