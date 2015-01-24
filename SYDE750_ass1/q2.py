import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d.axes3d import Axes3D
# Should I be using the dot product instead of multiply here?

def lif_neuron(x_inter, max_fire, t_ref=0.002, t_rc=0.02):
	beta = 1.0 / (1.0 - np.exp( (-1.0/max_fire + t_ref) / t_rc))
	alpha = (1.0 - beta)/(x_inter + 1.0)
	J_bias = 1.0 - alpha * x_inter
	print("beta: %s, alpha: %s, J_bias: %s" %(beta, alpha, J_bias))
	def lif(x):
		# because I can't figure out how to do the dot product with numpy
		J =  x * alpha + J_bias
		
		return_val = np.zeros(J.shape)
		# Select all the values where J > 1
		return_val[J > 1] += np.minimum(
					np.maximum(
						# Caluclate the activity
						1/(t_ref-t_rc*np.log(1-1/J[J > 1])),
						# make it zero if it's below zero
						np.zeros(return_val[J > 1].size)
					),
					# make it the max firing rate if it's above the max firing rate
					max_fire
				)
		# this should also return a scalar
		return return_val
	return lif

a = np.linspace(-1.0,1.0,100)
b = np.linspace(-1.0,1.0,100)

X,Y = np.meshgrid(a, b)

x_cept = 0.0
pref = np.array([1.0, -1.0])
pref = pref/np.linalg.norm(pref)
max_firing_rate = 100
neuron = lif_neuron(x_cept, max_firing_rate)
result = neuron( pref[0]*X + pref[1]*Y )

"""
ipdb.set_trace()
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, result, linewidth=0, cstride=1, rstride=1, cmap=plt.cm.jet)
plt.show()
"""

# grab the unit circle points
theta = np.linspace(0, 2*np.pi, 100)
xdata = np.array([np.cos(theta), np.sin(theta)])
ydata = neuron( np.dot(xdata.T, pref) )
"""
fig = plt.figure()
plt.plot([0,pref[0]], [0,pref[1]],'r')
plt.plot(xdata[0],xdata[1])
plt.show()
"""

"""
fig = plt.figure()
plt.plot(theta, ydata)
plt.plot([np.arctan2(pref[0],pref[1])],0,'rv')
plt.show()
"""

# fit the function
def cos_opt_func(x, A, B, C, D):
	return A*np.cos(B*x + C) + D

popt, _ = curve_fit(cos_opt_func, theta, ydata)

# compare
"""
fig = plt.figure()
plt.plot(theta, ydata)
plt.plot([np.arctan2(pref[0],pref[1])],0,'rv')
plt.plot(theta, cos_opt_func(theta, popt[0], popt[1], popt[2], popt[3]))
plt.show()
"""

# Cosine is a good choice given it's symmetrical nature. It's a poor choice due to it's inability to threshold around 0.

# 2_2

def gen_rand_uc_vecs(dims, number):
	vecs = np.random.normal(size=(number,dims))
	mags = np.linalg.norm(vecs, axis=-1)
	return vecs / mags[..., np.newaxis]

# generate random unit vectors and plot them
rand_vecs = gen_rand_uc_vecs(2, 100)

fig = plt.figure()
plt.plot(circle[0], circle[1])
for e in rand_vecs:
	plt.plot([0,e[0]], [0,e[1]], 'r')
plt.show()

x_vals = np.random.uniform(low=-1, size=(20, 2))