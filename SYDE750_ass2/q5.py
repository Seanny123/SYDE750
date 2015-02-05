import numpy as np
import ipdb
import matplotlib.pyplot as plt

# assuming we won't plug in a negative t
def ptsc(t, n, tau):
	#ipdb.set_trace()
	return ((t**n)*np.exp(-t/tau)) * (t > 0)

n_list = [0, 1, 2]
dt = 0.001
t_h = np.arange(1000)*dt-0.5
figure = plt.figure()
for n in n_list:
	res = ptsc(t_h, n, 0.007)
	res = res/np.linalg.norm(res)
	plt.plot(res, label="n=%s" %n)
plt.legend()
plt.savefig("5_a")

tau_list = [0.002, 0.005, 0.01, 0.02]
figure = plt.figure()
for tau in tau_list:
	res = ptsc(t_h, 0, tau)
	res = res/np.linalg.norm(res)
	plt.plot(res, label="tau=%s" %tau)
plt.legend()
plt.savefig("5_b")

