from abapy.materials import Hollomon
import matplotlib.pyplot as plt

E = 1.       # Young's modulus
sy = 0.1     # Yield stress
nu = 0.3     # Poisson's ratio
n = 0.3      # Hardening exponent
mat = Hollomon(labels = 'my_material', E=E, nu=nu, sy=sy, n=n)
eps_max = 1. # maximum strain to be computed
N = 10       # Number of points to be computed (10 is a low value useful for graphical reasons, in real simulations, 100 is a better value).
table = mat.get_table(0, N=N, eps_max=eps_max)
eps = table[:,0]
sigma = table[:,1]
sigma_max = max(sigma)

plt.figure()
plt.clf()
plt.title('Hollomon tensile behavior')
plt.xlabel('Strain $\epsilon$')
plt.ylabel('Stress $\sigma$')
plt.plot(eps, sigma, 'or-', label = 'Plasticity')
plt.plot([0., sy / E], [0., sy], 'b-', label = 'Elasticity')
plt.xticks([0., sy/E, eps_max], ['$0$', '$\epsilon_y$', '$\epsilon_{max}$'], fontsize = 16.)
plt.yticks([0., sy, sigma_max], ['$0$', '$\sigma_y$', '$\sigma_{max}$'], fontsize = 16.)
plt.grid()
plt.legend()
plt.show()
