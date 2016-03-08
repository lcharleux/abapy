import numpy as np
from abapy.indentation import ContactData

X = np.linspace(-3., 3., 512)
Y = np.linspace(-3., 3., 512)
X, Y = np.meshgrid(X, Y)
# Axi
cd = ContactData() 
x = [0, 1, 2, 10]
alt = [-1,.1, 0, 0]
press = [1, 0, 0, 0]
cd.add_data(x, altitude = alt, pressure = press)
Alt_axi, Press_axi = cd.interpolate(X, Y, method ='linear')
area = cd.contact_area()


# 3D
cd = ContactData(repeat = 3, is_3D = True)
k = np.cos(np.radians(60))
p = np.sin(np.radians(60))
x = [0, 4, 10, k*4, k*10]
y = [0, 0, 0,  p*4, p*10]
alt = [-1, 0, 0, 0, 0]
cd.add_data(x, altitude = alt)
Alt_3D, Press_3D = cd.interpolate(X, Y, method ='linear')


from matplotlib import pyplot as plt

fig = plt.figure()
plt.clf()
ax1 = fig.add_subplot(121)
grad = plt.imshow(Alt_axi)
plt.contour(Alt_axi, colors= 'black')
plt.title('axi contact')
ax1 = fig.add_subplot(122)
grad = plt.imshow(Alt_3D)
plt.contour(Alt_3D, colors= 'black')
plt.title('3D contact')
plt.show()
