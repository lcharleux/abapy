from abapy.postproc import HistoryOutput
from math import sin, pi
N = 100
hist = HistoryOutput()
time = [pi / 2 * float(i)/N for i in xrange(N+1)]
data = [sin(t) for t in time]
hist.add_step(time_step = time, data_step = data)
time2 = [10., 11.]
data2 = [1., 1.]
hist.add_step(time_step = time2, data_step = data2)
sol = 2. / pi + 1.
print 'Print computed value:', hist.average()
print 'Analytic solution:', sol
print 'Relative error: {0:.4}%'.format( (hist.average() - sol)/sol * 100.)
