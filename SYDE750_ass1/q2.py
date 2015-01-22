import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from mpl_toolkits.mplot3d.axes3d import Axes3D
# Should I be using the dot product instead of multiply here?

def lif_neuron(x_inter, max_fire, pref, t_ref=0.002, t_rc=0.02):
	beta = 1.0 / (1.0 - np.exp( (-1.0/max_fire + t_ref) / t_rc))
	# should alpha be a scalar? If so, how?
	alpha = (1.0 - beta)/(x_inter + 1.0)
	J_bias = 1.0 - alpha * x_inter
	def lif(x):
		# because I can't figure out how to do the dot product with numpy
		J =  pref[0]*x[0]+pref[1]*x[1] * alpha + J_bias
		
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

a = np.linspace(-1,1,50)
b = np.linspace(-1,1,50)

X,Y = np.meshgrid(a, b)

x_cept = 0.0
pref = np.array([-1.0, 1.0])
pref = pref/np.linalg.norm(pref)
max_firing_rate = 100
neuron = lif_neuron(x_cept, max_firing_rate, pref)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
# this certainly isn't passing in the right thing
ax.plot_surface(X, Y, neuron( [X, Y] ), 
                    linewidth=0, cstride=1, rstride=1, cmap=plt.cm.jet)
plt.show()

# grab the unit circle points
theta = np.linspace(0, 2*np.pi, 100)
xdata = np.array([np.cos(theta), np.sin(theta)])
ydata = neuron( np.dot(xdata.T, pref) )

# fit the function
def cos_opt_func(theta, A, B, C, D):
	return A*np.cos(B*theta + C) + D

popt, _ = scipy.optimize.curve_fit(cos_opt_func, xdata, ydata)
ipdb.set_trace()
# compare

# 2_2

def gen_rand_uc_vecs(dims, number):
	vecs = np.random.normal(size=(number,dims))
	mags = np.linalg.norm(vecs, axis=-1)
	return vecs / mags[..., np.newaxis]

# generate random unit vectors
e = gen_rand_uc_vecs(2, 100)

x_vals = np.random.uniform(low=-1, size=(20, 2))