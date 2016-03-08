
# Setting up the database
execfile('settings.py')

# Delaunay disp func
def disp(c1, c2):
  k = .1
  r = (c1**2 +c2**2)**.5
  r = r + (r==0.)
  c1 = c1 / r
  c2 = c2 / r
  theta = np.arccos(c1)
  theta += -2 * theta * (c2 < 0.)
  r = r * (1 + k* np.cos(theta * 3) / (1+k))
  return c1 * r, c2 * r


# creating some shortcuts
d = db_manager     # database manager
c = db_manager.cls # useful to call Simulation attributs 

simus2delete = d.query().filter(c.completed == False).all()

for s in simus2delete:
  d.session.delete(s)
d.session.commit()  

