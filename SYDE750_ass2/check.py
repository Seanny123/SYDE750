import numpy as np
import ipdb
import matplotlib.pyplot as plt
from utils import whitenoise, two_neurons, calc_rmse

# assuming we won't plug in a negative t
def ptsc(t, n, tau):
	return ((t**n)*np.exp(-t/tau)) * (t > 0)

dt = 0.001

tau_list = [0.002, 0.005, 0.01, 0.02]
t_h = np.arange(998)*dt-0.5
noise, _ = whitenoise(1, dt, 0.5, 5, 0)
for i_t, tau in enumerate(tau_list):
	h = ptsc(t_h, 0, tau=tau)
	neurons = two_neurons()
	res_noise = []
	for val in noise:
		res_noise.append(neurons(val))
	res_noise = np.array(res_noise)

	fspikes1 = np.convolve(res_noise[:,0], h, mode='same')
	fspikes2 = np.convolve(res_noise[:,1], h, mode='same')

	A = np.array([fspikes1, fspikes2]).T
	S = fspikes1.shape[0]

	gamma = np.dot(A.T, A) / S
	upsilon = np.dot(A.T, noise) / S
	decoders = np.dot(np.linalg.pinv(gamma), upsilon)
	x_hat = np.dot(A, decoders)

	# Plot x(t), the spikes, and x_hat
	figure = plt.figure()
	plt.plot(res_noise[:,0], label="neuron 1")
	plt.plot(res_noise[:,1], label="neuron 2")
	plt.plot(noise, label="signal")
	plt.plot(x_hat, label="approximation")
	plt.xlabel("time (s)")
	plt.legend()
	plt.savefig("check_tau=%s" %i_t)

n_list = [0, 1, 2]
for n_t, n_val in enumerate(n_list):
	h = ptsc(t_h, n_val, 0.007)
	neurons = two_neurons()
	res_noise = []
	for val in noise:
		res_noise.append(neurons(val))
	res_noise = np.array(res_noise)

	fspikes1 = np.convolve(res_noise[:,0], h, mode='same')
	fspikes2 = np.convolve(res_noise[:,1], h, mode='same')

	A = np.array([fspikes1, fspikes2]).T
	S = fspikes1.shape[0]

	gamma = np.dot(A.T, A) / S
	upsilon = np.dot(A.T, noise) / S
	decoders = np.dot(np.linalg.pinv(gamma), upsilon)
	x_hat = np.dot(A, decoders)

	# Plot x(t), the spikes, and x_hat
	figure = plt.figure()
	plt.plot(res_noise[:,0], label="neuron 1")
	plt.plot(res_noise[:,1], label="neuron 2")
	plt.plot(noise, label="signal")
	plt.plot(x_hat, label="approximation")
	plt.xlabel("time (s)")
	plt.legend()
	plt.savefig("check_n=%s" %n_t)