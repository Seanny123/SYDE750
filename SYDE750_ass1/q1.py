import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def rmse(predictions, targets):
	return np.sqrt(((predictions - targets) ** 2).mean())

def rec_lin_neuron(x_inter, max_fire, gain_sign, x_max=1.0):
	def rec_lin(x):
		return np.minimum(
				np.maximum(
					np.dot(gain_sign, x) * (max_fire/(x_max-x_inter))
					- x_inter * (max_fire/(x_max-x_inter)),
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
		J = np.dot(gain_sign, x) * alpha + J_bias
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

n_neurons = 16
rmse = []

max_firing_rates = np.random.uniform(100, 200, n_neurons)
x_cepts = np.random.uniform(-0.95, 0.95, n_neurons)
gain_signs = np.random.choice([-1, 1], n_neurons)


neurons = []
for i in range(n_neurons):
	neurons.append(rec_lin_neuron(x_cepts[i], max_firing_rates[i], gain_signs[i]))
x_vals = np.arange(-1, 1, 0.05)
S = x_vals.size # are we sure we're not supposed to use just x_vals?
A = np.zeros((n_neurons, x_vals.size))

fig = plt.figure()
for i in range(n_neurons):
	A[i,:] = neurons[i](x_vals)
	plt.plot(x_vals, A[i,:])
plt.grid('on')

def get_decoders(A, S):
	gamma = np.dot(A, A.T) / S
	upsilon = np.dot(A, x_vals) / S
	decoders = np.dot(np.linalg.inv(gamma), upsilon)
	x_hat = np.dot(A.T, decoders)
	return decoders, x_hat

decoders, x_hat = get_decoders(A, S)

# plot x_hat overlaid with x
fig = plt.figure()
plt.plot(x_vals, x_vals)
plt.plot(x_vals, x_hat)

# plot x_hat-x
fig = plt.figure()
plt.plot(x_vals, (x - x_hat))

rmse.append(predictions, targets)


A_noisy = A.T + numpy.random.normal(scale=0.2*numpy.max(A), size=A.shape)
xhat = numpy.dot(A_noisy, decoders)

# plot x_hat overlaid with x
fig = plt.figure()
plt.plot(x_vals, x_vals)
plt.plot(x_vals, x_hat)

# plot x_hat-x
fig = plt.figure()
plt.plot(x_vals, (x - x_hat))

rmse.append(predictions, targets)


decoders_noisy, x_hat_noisy = get_decoders(A, S)

# plot x_hat overlaid with x
fig = plt.figure()
plt.plot(x_vals, x_vals)
plt.plot(x_vals, x_hat)

# plot x_hat-x
fig = plt.figure()
plt.plot(x_vals, (x - x_hat))

rmse.append(predictions, targets)