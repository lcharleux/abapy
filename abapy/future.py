import numpy as np
import copy, math
from scipy.spatial import distance, Delaunay

def tri_area(vertices):
  u = vertices[0]
  v = vertices[1]
  w = vertices[2]
  return np.linalg.norm(np.cross( v-u, w-u)) / 2.
    
def tetra_volume(vertices):
  u = vertices[0]
  v = vertices[1]
  w = vertices[2]
  x = vertices[3]
  return np.cross(v-u, w-u).dot(x-u) / 6. 

class MasterDict(dict):
  """
  A dict with reverse reference to a master mesh.
  """
  def __init__(self, master, *args,**kwargs): 
    self.master = master
    dict.__init__(self,*args,**kwargs)

  def __setitem__(self, key, val):
    val.master = self.master
    dict.__setitem__(self, key, val)


#-------------------------------------------------------------------------------
# Fields

class Field(object):
  """
  A field meta class.
  """
  def __init__(self, label, loc, data = {}, mesh = None):
    self.label = label
    self.data = data
    self.master = master
    self.loc  = loc
    
  def toarrays(self):
    k = np.array(self.data.keys())
    v = np.array(self.data.values())
    return k, v   
    
class ScalarField(Field):
  """
  A scalar field class.
  """
  pass 

class VectorField(Field):
  """
  A vector field class.
  """
  pass 

class TensorField(Field):
  """
  A tensor field class.
  """
  pass 


#-------------------------------------------------------------------------------
# Nodes and related stuff


class Node(object):
  def __init__(self, coords, sets = set(), master = None):
    self.coords = np.asarray(coords, dtype= np.float64) 
    self.sets = set(sets)
    self.master = master
 
  def __repr__(self):
    return "#NODE" + self.__str__() 
  
  def __str__(self):
    return "{0}, sets={1}".format(self.coords, self.sets)

  
#-------------------------------------------------------------------------------
# Elements and related stuff    
    
class Element(object):
  
  _extrudable = False
  
  def __init__(self, conn, sets = set(), surfaces = None, master = None):
    self.conn = np.asarray(conn[:self._nvert], dtype = np.uint32)
    self.master = master
    self.sets = set(sets)
    ns = self.ns()
    self.surfaces = [set() for i in xrange(ns)]
    if surfaces != None:
      for k in surfaces.keys():
        self.surfaces[k].add( surfaces[k] ) 
    
  def __repr__(self):
    return "#ELEMENT " + self.__str__() 
    
  def __str__(self):
    name = self.type()
    conn = ",".join([str(l) for l in self.conn])
    return "{0}({1}), sets={2}, surf={3}".format(name, conn, 
                                              self.sets,
                                              self.surfaces)   
  
  def edges(self):
    """
    Returns the connectivity of the edges.
    """
    return self.conn[self._edges]
    
  def faces(self):
    """
    Returns the faces of a volumic element, None otherwise.
    """
    if self._space == 3:
      return np.array([ self.conn[f] for f in self._faces_conn])    
  
  def simplex_conn(self):
    """
    Returns the connectivity of the simplexes forming the element.
    """
    return np.array([ self.conn[f] for f in self._simplex_conn])
    
  def type(self):
    return self.__class__.__name__
  
  def ns(self):
    if self._space == 1: n = 1
    if self._space == 2: n = len(self._edges)
    if self._space == 3: n = len(self._faces_conn)
    return n
  
  def extrude(self, offset, layer):
    if self._extrudable:
      newtype = self._extrude_as
      oldconn = self.conn
      newconn = np.concatenate([
                oldconn + offset * layer, 
                oldconn + offset * (layer + 1)])[self._extrude_order]
      return globals()[newtype](conn = newconn, sets = self.sets)
  
  def collapsed(self):
    return len(self.conn) - len(set(self.conn))
  
  def collapsed_faces(self):
    if self._space == 3:
      faces = [self.conn[c] for c in self._faces_conn]
      return np.array([ len(f) - len(set(f)) for f in faces ])
    
  def collapsed_edges(self):
    if self._space  >= 2:
      edges = self.conn[self._edges]
      return np.array([ len(e) - len(set(e)) for e in edges ])  
 
  def simplex_decomposition(self):
    conn = self.conn
    if self._space == 3:
        simplices = self.simplex_conn()
        simplices2 = []
        for i in range(len(simplices)):
          simplex = simplices[i]
          if (len(set(simplex)) == 4):
            if tetra_volume([self.master.nodes[l].coords for l in simplex]) > 0.:
              simplices2.append(simplices[i])
          
        return [Tetra4(simplex) for simplex in simplices2]
        """
        points = [self.master.nodes[l].coords for l in conn]
        tetras = Delaunay(points).simplices
        tetras2 = []
        for i in xrange(len(tetras)): 
          if tetra_volume([points[j] for j in tetras[i]]) < 0.:
            t = np.array([tetras[i][j] for j in [1, 0, 2, 3]])
          else: 
            t = np.array([tetras[i][j] for j in [0, 1, 2, 3]])  
        return [Tetra4(conn[t]) for t in tetras]
        """
        
    if self._space == 2:
      if self.type() == "Tri3":
        if len(conn) == 3: 
          return [self]
      if self.type() == "Quad4":
        if len(conn) == 4: 
           return [Tri3(self.conn[c]) for c in [[0, 1, 2], [1, 2, 3]]]    
        if len(conn) == 3:
          count = np.array([(self.conn == i).sum() for i in conn])
          rind = conn[np.where(count == 2)[0][0]]
          rconn = self.conn.copy()
          for i in range(4):
            if (rconn[1] == rind) and (rconn[-1] == rind):
              return [Tri3(rconn[[0,1,2]])]
            rconn = np.roll(rconn, 1)
                    
        
  def clean_connectivity(self):
    if self.collapsed() == 0:
      return [self]
    else:
      return self.simplex_decomposition()              
  
  def node_set_to_surface(self, nodesetlabel, surfacelabel):
    nodelabels = set([k for k in self.nodes.keys() if label in self.nodes[k].sets])
    for element in self.elements.values:
      for i in xrange(self.ns()):
        if self._space == 3: surfconn = self.conn[self._faces_conn[i]]
        # TO be completed
        if nodelabels.issuperset(surfconn):
          self.surfaces[i].add(surfacelabel)
  
  def volume(self, add = True):
    vertices = np.array([self.master.nodes[l].coords for l in self.conn ])
    simplices = vertices[self._simplex_conn]
    if self._space == 3:
      v = np.array([tetra_volume(simplex) for simplex in simplices])
      if add: v = v.sum()
      return v
    if self._space == 2:
      v = np.array([tri_area(simplex) for simplex in simplices])
      if add: v = v.sum()
      return v   
  
  def centroid(self):
    vertices = np.array([self.master.nodes[l].coords for l in self.conn ])
    simplices = vertices[self._simplex_conn]
    centroids = simplices.mean(axis = 1)
    volumes = self.volume(add = False)[:,None]
    return (centroids * volumes).sum(axis = 0) / volumes.sum()
  
          
    
class Line2(Element):
  """
  A 1D 2 nodes line.
  """    
  _nvert = 2
  _space = 1
  _extrudable = True
  _extrude_as = "Quad4"
  _extrude_order = np.array([0, 1, 3, 2])
  _simplex_conn = np.array([[0, 1]])
  
  
class Tri3(Element):
  """
  A 2D 3 noded triangular element
  """
  _nvert  = 3
  _space  = 2
  _edges = np.array([[0, 1],
                     [1, 2],
                     [2, 0]])
  _extrudable = True                        
  _extrude_as = "Prism6"
  _extrude_order = np.array([0, 1, 2, 3, 4, 5])
  _simplex_conn  = np.array([[0, 1, 2]])

class Quad4(Element):
  """
  A 2D 4 noded quadrangular element
  """
  _nvert  = 4
  _space  = 2 
  _edges = np.array([[0, 1],
                     [1, 2],
                     [2, 3],
                     [3, 0]])
  _extrudable = True
  _extrude_as = "Hexa8"
  _extrude_order = np.array([0, 1, 2, 3, 4, 5, 6, 7])
  _simplex_conn  = np.array([[0, 1, 3], 
                             [1, 2, 3]])                        

class Tetra4(Element):
  """
  A 3D 4 noded tetrahedral element
  """
  _nvert  = 4
  _space  = 3
  _faces_conn  = np.array([[0, 1, 2],
                           [0, 3, 1],
                           [1, 3, 2],
                           [2, 3, 0]])
  _faces_type  = ["Tri3", "Tri3", "Tri3", "Tri3"]                        
  _edges = np.array([[0, 1],
                     [1, 2],
                     [2, 0],
                     [0, 3],
                     [1, 3],
                     [2, 3]])
  _simplex_conn  = np.array([[0, 1, 3, 4]])
                          
  def clean_connectivity(self):
    if self.collapsed():
      return None
    else:
      return [self]  
  

class Pyra5(Element):
  """
  A 3D 5 noded pyramidal element
  """
  _nvert  = 5
  _space  = 3
  _faces_conn  = np.array([[0, 1, 2, 3],
                           [0, 1, 4],
                           [1, 2, 4],
                           [2, 3, 4],
                           [3, 0, 4]])
  _faces_type = ["Quad4", "Tri3", "Tri3", "Tri3", "Tri3"]                        
  _edges = np.array([[0, 1],
                     [1, 2],
                     [2, 3],
                     [3, 0],
                     [0, 4],
                     [1, 4],
                     [2, 4],
                     [3, 4]])
  _simplex_conn  = np.array([[0, 1, 3, 4],
                             [1, 2, 3, 4]])
  
class Prism6(Element):
  """
  A 3D 6 noded prismatic element
  """
  _nvert  = 6
  _space  = 3  
  _faces_conn = np.array([[0, 1, 2],
                          [3, 5, 4],
                          [0, 3, 4, 1],
                          [1, 4, 5, 2],
                          [2, 5, 3, 0]])
  _faces_type = ["Tri3", "Tri3", "Quad4", "Quad4", "Quad4"]                        
  _edges = np.array([[0, 1],
                     [1, 2],
                     [2, 0],
                     [3, 4],
                     [4, 5],
                     [5, 3],
                     [0, 3],
                     [1, 4],
                     [2, 5]])
  _simplex_conn  = np.array([[0, 1, 2, 3],
                             [1, 2, 3, 4],
                             [2, 3, 4, 5]])                
                     
class Hexa8(Element):
  """
  A 3D 8 noded hexahedric element
  """
  _nvert  = 8
  _space  = 3  
  _faces_conn  = np.array([[0, 1, 2, 3],
                           [4, 7, 6, 5],
                           [0, 4, 5, 1],
                           [1, 5, 6, 2],
                           [2, 6, 7, 3],
                           [3, 7, 4, 0]])
  _faces_type = ["Quad4",
                 "Quad4",
                 "Quad4",
                 "Quad4",
                 "Quad4",
                 "Quad4"]                        
  _edges = np.array([[0, 1],
                     [1, 2],
                     [2, 3],
                     [3, 0],
                     [4, 5],
                     [5, 6],
                     [6, 7],
                     [7, 4],
                     [0, 4],
                     [1, 5],
                     [2, 6],
                     [3, 7]])
  _simplex_conn = np.array([[0, 1, 3, 4],
                            [1, 2, 3, 4],
                            [3, 2, 7, 4],  
                            [2, 6, 7, 4],
                            [1, 5, 2, 4],
                            [2, 5, 6, 4]])
#-------------------------------------------------------------------------------       
# Mesh & related stuff
class Mesh(object):
  def __init__(self, nodes = {}, elements = {}):
    self.nodes    = MasterDict(self)
    for k, v in nodes.iteritems(): self.nodes[k] = v
    self.elements = MasterDict(self)
    for k, v in elements.iteritems(): self.elements[k] = v
      
  def __repr__(self):
    nn = len(self.nodes.values())
    ne = len(self.elements.values())
    return "#MESH: {0} nodes / {1} elements".format(nn, ne)
  
  def __str__(self):
    nodes, elements = self.nodes, self.elements
    nk = sorted(nodes.keys())
    ns = "\n".join( ["{0} {1}".format(k, str(nodes[k])) for k in nk])
    ek = sorted(elements.keys())
    es = "\n".join( ["{0} {1}".format(k, str(elements[k])) for k in ek])
    """
    nsets, esets = self.nsets, self.esets
    nsk = sorted(nsets.keys())
    nss = "\n".join( ["{0} {1}".format(k, str(nsets[k])) for k in nsk])
    esk = sorted(esets.keys())
    print esk
    ess = "\n".join( ["{0} {1}".format(k, str(esets[k])) for k in esk])
    """
    return "MESH:\n*NODES:\n{0}\n*ELEMENTS:\n{1}".format(ns, es)
  """
  def _add_set(self, kind, key, labels):
    if kind == "nset": target = self.nodes
    if kind == "eset": target = self.elements
    for label in labels:
      target[label].sets.add(key)
  
  def add_nset(self, *args, **kwargs):
    self._add_set(kind = "nset", *args, **kwargs)
  
  def add_eset(self, *args, **kwargs):
    self._add_set(kind = "eset", *args, **kwargs)
  """
  def export(path):
    return  
  
  def load(path):
    return
  
  def extrude(self, translation, layers):
    translation = np.array(translation, dtype = np.float64)[:3]
    newmesh = Mesh()
    # Nodes:
    node_offset = max(self.nodes.keys())
    for l, n in self.nodes.iteritems():
      for j in xrange(layers+1):
        newnode = Node(coords = n.coords + translation * float(j) / layers,                 
                       sets = n.sets)
        newmesh.nodes[l + j * node_offset] = newnode
    # Elements:
    element_offset = max(self.elements.keys())
    for l, e in self.elements.iteritems():
      for layer in xrange(layers):
        newelement = e.extrude(offset = node_offset, layer = layer)
        if newelement != None:
          newmesh.elements[l + layer * element_offset] = newelement 
    return newmesh
  
  def copy(self):
    return copy.deepcopy(self)
  
  def nodes_to_array(self):
    labels = np.array([k for k in self.nodes.keys()])
    n = len(labels)
    p = np.empty((n,3))
    for i in range(n): 
      p[i] = self.nodes[labels[i]].coords
    return labels , p
    
    
  def transform(self, transformation):
    labels, p  = self.nodes_to_array()
    n = len(labels)
    x, y, z = p.transpose()
    newcoords = np.asarray(transformation(x, y, z), 
                           dtype = np.float64).transpose()
    newmesh = self.copy()
    for i in range(n): 
      newmesh.nodes[labels[i]].coords = newcoords[i]  
    return newmesh
    
  def overlapping_nodes(self, crit_dist = 1.e-6):
    def calc_row_idx(k, n):
      return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

    def elem_in_i_rows(i, n):
      return i * (n - 1 - i) + (i*(i + 1))/2

    def calc_col_idx(k, i, n):
      return int(n - elem_in_i_rows(i + 1, n) + k)

    def condensed_to_square(k, n):
      i = calc_row_idx(k, n)
      j = calc_col_idx(k, i, n)
      return np.array([i, j])
    labels, points = self.nodes_to_array()
    dist = distance.pdist(points)
    n = len(labels)
    loc = np.where(dist<=crit_dist)[0]
    pairs = [labels[condensed_to_square(l, n)] for l in loc]
    mapping = dict(zip(labels, labels))
    for pair in pairs:
      pair.sort()
      mapping[pair[1]] = min(mapping[pair[1]],  pair[0])
    return mapping   
    
  def merge_nodes(self, mapping):
    newmesh = self.copy()
    for elem in newmesh.elements.values():
        conn = elem.conn
        for i in range(len(conn)):
          conn[i] = mapping[conn[i]]
    
    for label in newmesh.nodes.keys():
      if mapping[label] != label:
        del newmesh.nodes[label]
    return newmesh            
  
  def simplex_decomposition(self):
    """
    Returns a list of new simplex elements sharing the same vertices with the 
    orginal one with a viable connectivity.
    """
    newmesh = self.copy()
    newmesh.elements.clear()
    label = 1
    for el in self.elements.values():
      simplices = el.simplex_decomposition()   
      if simplices != None:
        for simplex in simplices:
          newmesh.elements[label] = simplex
          label += 1
    return newmesh                              
  
  def clean_connectivity(self):
    newmesh = self.copy()
    newmesh.elements.clear()
    label = 1
    for el in self.elements.values():
      newels = el.clean_connectivity()   
      if newels != None:
        for newel in newels:
          newmesh.elements[label] = newel
          label += 1
    return newmesh           
#-------------------------------------------------------------------------------
# Parsers & writers

# Abaqus INP

def parseInp(path):
  # Some useful tools
  def lineInfo(line):
    out =  {"type": "data"}
    if line[0] == "*":
      if line[1] == "*": 
        out["type"] = "comment"
        out["text"] = line[2:]
      else:
        out["type"] = "command"
        words = line[1:].split(",")
        out["value"] = words[0].strip()
        out["options"] = {}
        for word in words[1:]:
          key, value =  [s.strip() for s in word.split("=")]
          out["options"][key] = value
    return out
  def elementMapper(inpeltype):
    if inpeltype == "t3d2": return "Line2"
    if inpeltype[:3] in ["cps", "cpe", "cax"]:
      if inpeltype[3] == "3": return "Tri3"
      if inpeltype[3] == "4": return "Quad4"
    if inpeltype[:3] in ["c3d"]:
      if inpeltype[3] == "4": return "Tetra4"
      if inpeltype[3] == "5": return "Pyra5"
      if inpeltype[3] == "6": return "Prism6"
      if inpeltype[3] == "8": return "Hexa8"
    
      
  # Output mesh
  m = Mesh()
  # File preprocessing
  lines = np.array([l.strip().lower() for l in open(path).readlines()])
  # Data processing
  env = None
  setlabel = None
  for line in lines: 
    d = lineInfo(line)
    if d["type"] == "command": 
      env = d["value"]
      # Nodes
      if env == "node":
        opt = d["options"]
        currentset = None
        if "nset" in opt.keys(): currentset = opt["nset"]
          
      # Elements
      if env == "element":
        opt = d["options"]
        eltype = elementMapper(opt["type"])
        currentset = None
        if "elset" in opt.keys(): currentset = opt["elset"]
          
      # Nsets
      if env == "nset":
        opt = d["options"]      
        currentset = opt["nset"]
        
      # Elsets     
      if env == "elset":
        opt = d["options"]      
        currentset = opt["elset"]
             
    if d["type"] == "data": 
      words = line.strip().split(",")
      if env == "node": 
        label  = int(words[0])
        coords = np.array( [float(w) for w in words[1:]], dtype = np.float64 )
        if currentset == None: 
          m.nodes[label] = Node(coords = coords)
        else:
          m.nodes[label] = Node(coords = coords, sets = set([currentset]))  
            
      if env == "element": 
        label  = int(words[0])
        conn = np.array( [int(w) for w in words[1:]], dtype = np.int32)
        if currentset == None: 
          m.elements[label] = globals()[eltype](conn = conn)
        else:
          m.elements[label] = globals()[eltype](conn = conn, sets = set([currentset]))
      
      if env == "nset": 
        [m.nodes[int(w)].sets.add(currentset) for w in words if len(w) != 0]   
        
      if env == "elset": 
        [m.elements[int(w)].sets.add(currentset) for w in words if len(w) != 0]         
        
              
  return m
  
def writeInp(mesh, mapping, path = None):
  
  def exportset(s, d):
    out = ""
    labels = [str(k) for k,v in d.iteritems() if s in v.sets]
    for i in xrange(len(labels)):
      out += labels[i]
      if (i+1)%10 != 0:
        out += ", "
      else:
        out += ",\n"
    if out[-1] != "\n": out += "\n"
    return out
    
  # Nodes
  out = "*NODE\n"
  for label, node in mesh.nodes.iteritems():
    out += "{0}, {1}\n".format(label, ", ".join([ str(c) for c in node.coords]))
  # Elements
  etypes = set([e.type() for e in mesh.elements.values()])
  for etype in etypes:
    out +="*ELEMENT, TYPE={0}\n".format( mapping[etype])
    for label, elem in mesh.elements.iteritems():
      if elem.type() == etype:
        out += "{0}, {1}\n".format(label, ", ".join([ str(c) for c in elem.conn]))
  # Sets
  nsets = set().union(*[n.sets for n in mesh.nodes.values()])
  for s in nsets: 
    out += "*NSET, NSET={0}\n".format(s) + exportset(s , mesh.nodes)
  esets = set().union(*[e.sets for e in mesh.elements.values()])
  for s in esets:
    out += "*ELSET, ELSET={0}\n".format(s) + exportset(s , mesh.elements)
  if path == None:
    return out
  else:
    open(path, "w").write(out)  


def writeMsh(mesh, path = None):
  elementMap = {"Tri3": 2, "Quad4":3, "Tetra4":4, "Hexa8":5, "Prism6":6, "Pyra5": 7}
  pattern = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
{0}
$EndNodes
$Elements
{1}
$EndElements"""
  nodeout = "{0}\n".format(len(mesh.nodes.keys()))
  nodelist = []
  for k in mesh.nodes.keys():
    node = mesh.nodes[k]
    coords = node.coords
    nodelist.append("{0} {1} {2} {3}".format(k, coords[0], coords[1], coords[2]) )
  nodeout += "\n".join(nodelist)
  elemout = ""
  elemout = "{0}\n".format(len(mesh.elements.keys()))
  elemlist = []
  for k in mesh.elements.keys():
    element = mesh.elements[k]
    coords = node.coords
    elemlist.append("{0} {1} 1 1 {2}".format(
        k, 
        elementMap[element.__class__.__name__],
        " ".join( [str(l) for l in element.conn ] ) ))
  elemout += "\n".join(elemlist)
  if path == None:
    return pattern.format(nodeout, elemout)
  else:
    open(path, "w").write(pattern.format(nodeout, elemout))  
               
