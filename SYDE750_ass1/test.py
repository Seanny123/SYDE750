import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def rec_lin_neuron(x_inter, max_fire, gain_sign):
	print("x_inter:%s, max_fire:%s" %(x_inter, max_fire))
	def rec_lin(x):
		return np.minimum(
				np.maximum(
					gain_sign * (max_fire/(1.0-x_inter)) * x
					- x_inter * (max_fire/(1.0-x_inter)),
					np.zeros(x.size)
				),
				max_fire
			)
	return rec_lin

def lif_neuron(x_inter, max_fire, gain_sign, t_ref=0.002, t_rc=0.02):
	beta = 1.0 / (1.0 - np.exp( (-1.0/max_fire + t_ref) / t_rc))
	alpha = (1.0 - beta)/(x_inter + 1.0)
	J_bias = 1.0 - alpha * x_inter
	def lif(x):
		J = gain_sign * alpha * x + J_bias
		return_val = np.zeros(x.shape[0])
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
		return return_val
	return lif

"""
def lol_neuron():
	def lol(x):
		J = x + 1
		J[J > 1] += 1
		return J
	return lol
"""

x_vals = np.arange(0.8, 1.5, 0.05)
x_cept = 0.95
gain_sign = 1
max_firing_rate = 120
neuron = lif_neuron(x_cept, max_firing_rate, gain_sign)
print(neuron(x_vals))
ipdb.set_trace()
plt.plot(x_vals, neuron(x_vals))
plt.grid('on')
plt.show()
ipdb.set_trace()