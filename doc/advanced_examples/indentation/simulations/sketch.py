from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

# Setting up the database
execfile('settings.py')

# creating some shortcuts
d = db_manager     # database manager
c = db_manager.cls # useful to call Simulation attributs 

# Chosing the simulation
'''
simus = d.query().filter(c.completed == True).all()
simus = [s for s in simus 
  if s.sample_mat_type == 'druckerprager'
  and s.sample_mat_args['yield_stress'] == 0.015
  and s.sample_mat_args['beta'] == 15.]
simu = simus[0]
'''
s = d.query().filter(c.id == 12).first()

# Force vs Disp curve
F_hist = s.force_hist
D_hist = s.disp_hist
trash, F_l = F_hist[0:2].toArray()
trash, D_l = D_hist[0:2].toArray()
F_l, D_l = np.array(F_l), np.array(D_l)
trash, F_u = F_hist[2].toArray()
trash, D_u = D_hist[2].toArray()
F_u, D_u = np.array(F_u), np.array(D_u)
d_m = D_l.max()
f_m = s.load_prefactor * d_m**2
F_u = F_u[np.where(F_u > f_m * 1.e-6)]
D_u = D_u[np.where(F_u > f_m * 1.e-6)]
d_f = s.final_displacement
F_l_fit = s.load_prefactor * D_l**2
F_u_fit = f_m * ((D_u - d_f) / (d_m - d_f))**s.unload_exponent
stiff = s.contact_stiffness *f_m /d_m
S_u = (D_u - d_m) * stiff + f_m
D_us = D_u[np.where(S_u > f_m * 1.e-6)]
S_u = S_u[np.where(S_u > f_m * 1.e-6)]


fs = 14.
lw = 2.
plt.figure(0)
plt.clf()
#plt.xticks([0., d_f, d_m, d_m - f_m/stiff], ['$0$', '$d_f$', '$d_m$', r'$d_m - \frac{F_m}{S}$'], fontsize = fs)
plt.xticks([0., d_f, d_m], ['$0$', '$d_f$', '$d_m$'], fontsize = fs)
plt.yticks([0., f_m], ['$0$', '$F_m$'], fontsize = fs)
plt.xlabel('Displacement $d$', fontsize = fs)
plt.ylabel('Force $F$', fontsize = fs)
plt.grid()
plt.plot(D_l, F_l, '+r', label = 'Loading data')
plt.plot(D_l, F_l_fit, 'r-', label = r'Loading fit: $F = C d^2$ ', linewidth = lw)
plt.plot(D_u, F_u, '+b', label = 'Unloading')
plt.plot(D_u, F_u_fit, 'b-', label = r'Unloading fit: $F = F_m \left(\frac{d - d_f}{d_m-d_f}\right)^m$', linewidth = lw)
plt.plot(D_us, S_u, 'g-', label = r'Unloading fit tangent', linewidth = 1.)
plt.legend(loc = 'upper left')
plt.savefig('plots/sketch_load_disp.pdf')

# Sections:
lw = 2.
r, N = 4., 1024
z_min, z_max = -1., .2
R = np.linspace(0., r, N)
CD_l = s.contact_data[1][-1]
CD_u = s.contact_data[2][-1]
Alt_l, Press_l = CD_l.interpolate(R, np.zeros_like(R), method = 'linear')
Alt_u, trash = CD_u.interpolate(R, np.zeros_like(R), method = 'linear')
alt_ml = max(0., z_max)
alt_mu = max(0., z_max)
phi = s.indenter_half_angle
r_c = ((s.contact_area * d_m**2)/np.pi)**.5
Ind_l = R/np.tan(np.radians(phi)) - d_m
R_ind_l = R[np.where(Ind_l <=alt_ml)]
Ind_l = Ind_l[np.where(Ind_l <=alt_ml)]
Ind_u = R/np.tan(np.radians(phi)) + Alt_u.min()
R_ind_u = R[np.where(Ind_u <=alt_mu)]
Ind_u = Ind_u[np.where(Ind_u <=alt_mu)]
Ind_v = R/np.tan(np.radians(phi)) + Alt_u.min() - Alt_u
R_ind_v = R
fig = plt.figure(0)
plt.clf()
ax = fig.add_subplot(311)
plt.grid()
ax.set_title('Virtual Indenter')
#plt.xlabel('$r$', fontsize = fs)
plt.ylabel('$z$', fontsize = fs)
plt.plot(R / d_m,Alt_l/d_m, '-b', label = 'Sample$_{h=h_m}$', linewidth = lw)
plt.plot(R_ind_l / d_m, Ind_l / d_m , '-r', label = 'Indenter$_{h=h_m}$', linewidth = lw)
plt.xticks([0., r_c], ['', '$r_c$'], fontsize = fs)
plt.yticks([-1., 0.], ['$d_m$', '$0$'], fontsize = fs)
plt.legend(loc = 'upper left')
plt.ylim([z_min, z_max])
ax = fig.add_subplot(312)
plt.grid()
#ax.set_title('Unloading: $h=h_f$', fontsize = fs)
#plt.xlabel('$r$', fontsize = fs)
plt.ylabel('$z$', fontsize = fs)
plt.plot(R / d_m,Alt_u/d_m, '-b', label = 'Sample$_{h=h_f}$', linewidth = lw)
plt.plot(R_ind_u / d_m, Ind_u / d_m , '-r', label = 'Indenter$_{h=h_f}$', linewidth = lw)
plt.xticks([0., r_c], ['', '$r_c$'], fontsize = fs)
plt.yticks([-1., -d_f/d_m, 0.], ['$d_m$', r'$d_f$' ,'0'], fontsize = fs)
plt.legend(loc = 'upper left')
plt.ylim([z_min, z_max])
ax = fig.add_subplot(313)
plt.grid()
#ax.set_title('Virtual indenter', fontsize = fs)
plt.xlabel('$r$', fontsize = fs)
plt.ylabel('$z$', fontsize = fs)
plt.plot(R_ind_v / d_m, Ind_v / d_m , '-g',  linewidth = lw, label = 'Virtual Indenter = Indenter$_{h=h_f}$ - Sample$_{h=h_f}$')
d_u = d_m - d_f
r_ind = interpolate.interp1d(Ind_v/d_m, R_ind_v/d_m) # Interpolation function of the radius of the virtual indenter
r_c_me = r_ind(.5 *d_u)
plt.xticks([0., r_c, r_c_me], ['$0$', '', '$r_{c}^*$'], fontsize = fs)
plt.yticks([0., .5*d_u], ['$0$',r'$\frac{d_m - d_f}{2}$' ], fontsize = fs)
plt.legend(loc = 'upper left')
plt.savefig('plots/sketch_profiles.pdf')
