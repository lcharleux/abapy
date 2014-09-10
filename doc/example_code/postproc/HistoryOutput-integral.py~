from abapy.postproc import HistoryOutput
time = [ [0., 1.], [3., 4.] ]
data = [ [.5, 1.5], [.5, 1.5] ]
hist = HistoryOutput(time = time, data = data)
hist[0].integral()
hist[1].integral()
hist.integral()
N = 10
from math import sin, pi
time = [pi / 2 * float(i)/N for i in xrange(N+1)]
data = [sin(t) for t in time]
hist = HistoryOutput()
hist.add_step(time_step = time, data_step = data)
trap = hist.integral()
simp = hist.integral(method = 'simps')
trap_error = (trap -1.)
simp_error = (simp -1.)
print 'Relative errors:\nTrapezoid rule: {0:.2}%\nSimpson rule: {1:.2}%'.format(trap_error*100, simp_error*100)
