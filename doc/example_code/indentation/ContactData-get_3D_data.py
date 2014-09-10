from abapy.indentation import ContactData
from matplotlib import pyplot as plt


# Axi contact
cd = ContactData() 
x = [0, 1, 2, 3]
alt = [-1,.1, 0, 0]
press = [1, 0, 0, 0]
cd.add_data(x, altitude = alt, pressure = press)
points_axi, alt_axi, press_axi, conn_axi = cd.get_3D_data(axi_repeat = 20)



# 3D contact
cd = ContactData(repeat = 3, is_3D = True)
k = np.cos(np.radians(60))
p = np.sin(np.radians(60))
x = [0, 4, 10, k*4, k*10]
y = [0, 0, 0,  p*4, p*10]
alt = [-1, 0, 0, 0, 0]
cd.add_data(x, altitude = alt)
points_3D, alt_3D, press_3D, conn_3D = cd.get_3D_data()



fig = plt.figure()
plt.clf()
ax1 = fig.add_subplot(121)
ax1.set_aspect('equal')
plt.tricontourf(points_axi[:,0], points_axi[:,1], conn_axi, alt_axi)
plt.triplot(points_axi[:,0], points_axi[:,1], conn_axi)

plt.title('axi contact')
ax1 = fig.add_subplot(122)
ax1.set_aspect('equal')
plt.tricontourf(points_3D[:,0], points_3D[:,1], conn_3D, alt_3D)
plt.triplot(points_3D[:,0], points_3D[:,1], conn_3D)
plt.title('3D contact')
plt.show()
