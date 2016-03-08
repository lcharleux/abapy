import matplotlib.pyplot as plt
from abapy.postproc import HistoryOutput
time = [ [1., 2.,3.] , [3.,4.,5.] , [5.,6.,7.] ]
force = [ [2.,2.,2.] , [3.,3.,3.] , [4.,4.,4.] ]
Force = HistoryOutput(time, force)
fig = plt.figure(0, figsize=(8,4))
plt.clf()
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
x,y = Force[[0,2]].toArray()
x2,y2 = Force[[0,2]].plotable()
ax.plot(x,y)
ax2.plot(x2,y2)
ax.set_ylim([1,5])
ax2.set_ylim([1,5])
plt.savefig('HistoryOutput-plotable.png')
ax.set_title('HistorytOutput.toArray')
ax2.set_title('HistorytOutput.plotable')
plt.show()
