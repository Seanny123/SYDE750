import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def calc_rmse(predictions, targets):
	return np.sqrt( (np.square(predictions - targets)).mean() )

def drms(sig):
	return np.sqrt(np.sum(np.square(sig))/sig.size)

def lif_neuron(x_inter, max_fire, t_ref=0.002, t_rc=0.02, radius=1.0):
	beta = 1.0 / (
		1.0 - np.exp(
			(-1.0/max_fire + t_ref) / t_rc
		)
	)
	# okay, how the hell do you take the radius into account?
	alpha = 1.0/radius * (1.0 - beta)/(x_inter + 1.0)
	#J_bias = 1.0 - alpha * x_inter * radius
	#alpha = (1.0 - beta)/(x_inter + 1.0)
	J_bias = 1.0 - alpha * x_inter
	def lif(x):
		J = x * alpha + J_bias
		return_val = np.zeros(x.shape[0])
		# Select all the values where J > 1
		return_val[J > 1] += np.maximum(
						# Caluclate the activity
						1.0/(t_ref-t_rc*np.log(1-1/J[J > 1])),
						# make it zero if it's below zero
						np.zeros(return_val[J > 1].size)
					)
		return return_val
	return lif

def modified_lif(x0_fire, max_fire, t_ref=0.002, t_rc=0.02):
	beta = 1.0 / (
		1.0 - np.exp(
			(-1.0/max_fire + t_ref) / t_rc
		)
	)
	J_bias = 1.0 / (
		1.0 - np.exp(
			(-1.0/x0_fire + t_ref) / t_rc
		)
	)
	alpha = beta - J_bias
	def lif_current(x):
		J = x * alpha + J_bias
		if(J > 1):
			return J
		else:
			return 0.0

	return lif_current

def additive_lif(x_inter, max_fire, t_ref=0.002, t_rc=0.02):
	beta = 1.0 / (
		1.0 - np.exp(
			(-1.0/max_fire + t_ref) / t_rc
		)
	)
	alpha = (1.0 - beta)/(x_inter + 1.0)
	J_bias = 1.0 - alpha * x_inter
	def lif(x):
		# dot product with the decoders?
		J = x * alpha + J_bias
		return_val = np.zeros(x.shape[0])
		# Select all the values where J > 1
		return_val[J > 1] += np.maximum(
						# Caluclate the activity
						1.0/(t_ref-t_rc*np.log(1-1/J[J > 1])),
						# make it zero if it's below zero
						np.zeros(return_val[J > 1].size)
					)
		return return_val
	return lif

def get_activities(neuron_type, x_vals, n_neurons, x_cepts, max_firing_rates, gain_signs, radius=1.0):
	neuron_list = []
	A = np.zeros((n_neurons, x_vals.size))
	for i in range(n_neurons):
		neuron_list.append(neuron_type(x_cepts[i], max_firing_rates[i], radius=radius))
		A[i,:] = neuron_list[i](x_vals*gain_signs[i])
	return A, neuron_list

def get_decoders(A, S, x_vals):
	gamma = np.dot(A, A.T) / S
	upsilon = np.dot(A, x_vals) / S
	decoders = np.dot(np.linalg.pinv(gamma), upsilon)
	x_hat = np.dot(A.T, decoders)
	return decoders, x_hat

def plot_xhat(x_vals, x_hat, title, filename):
	# plot x_hat overlaid with x
	fig = plt.figure()
	plt.plot(x_vals, x_vals, label="real value")
	plt.plot(x_vals, x_hat, label="approximated value")
	plt.title("x_hat %s" %title)
	plt.xlabel("x")
	plt.ylabel("Firing Rate (Hz)")
	plt.xlim([-1,1])
	plt.legend(loc=4)
	plt.savefig("%s_1" %filename)

	# plot x_hat-x
	fig = plt.figure()
	plt.plot(x_vals, (x_vals - x_hat))
	plt.title("approximation error %s" %title)
	plt.xlabel("x")
	plt.ylabel("Error")
	plt.xlim([-1,1])
	plt.savefig("%s_2" %filename)

def whitenoise(T, dt, rms, limit, seed):
	# randomly generate co-efficient from a gaussian distribution equal to half the size of frequencies
	# 1/dt / 1/T / 2
	np.random.seed(seed)
	coef = np.random.normal(0, 1,
					int( np.ceil(T/(2.0*dt)) )
				) + 1j * np.random.normal(0, 1,
					int( np.ceil(T/(2.0*dt)) )
				)

	# eliminate anything over the limit
	frequencies = np.arange(0, coef.size, 1.0)
	coef[frequencies > limit] = 0.0
	coef[0] = 0.0
	if(coef.size % 2 == 1):
		print("odd")
		final_coef = np.zeros(coef[1:].size * 2 + 1, dtype=np.complex_)
		final_coef[coef.size] = coef
		final_coef[coef.size:] = coef[1:][::-1].conj()
	else:
		final_coef = np.zeros(coef[1:-1].size * 2 + 2, dtype=np.complex_)
		# Don't touch the DC or the middle term!
		final_coef[:coef.size] = coef
		final_coef[coef.size:] = coef[1:-1][::-1].conj()

	# bring it to the time domain
	time_domain = np.fft.ifft(final_coef)
	# make it totally real
	time_domain = np.real(time_domain)
	# calculate the rms and scale it
	curr_rms = drms(time_domain)
	time_domain = (rms/curr_rms)*time_domain
	return time_domain, (rms/curr_rms)*final_coef

def spiking(potential):
	if(potential < 2):
		return 0.0
	else:
		return 1.0

class SpikingLif(object):
	def __init__(self, t_ref=0.002, t_rc=0.02, dt=0.001):
		self.t_ref = t_ref
		self.t_rc = t_rc
		self.refac_time = 0.0
		self.refac = True
		self.dt = dt
		self.potential = 0.0
		self.spike_count = 0

	def spike(self, current):
		if(self.refac == False):
			# use that differential equation
			dV = 1.0/self.t_rc*(current-self.potential)*self.dt
			#ipdb.set_trace()
			self.potential += dV
			# If we've reached 1, spike
			if(self.potential >= 1):
				# start the refactory period and reset the potential
				self.refac = True
				self.potential = 0.0
				self.spike_count += 1
				return 2.0
			return self.potential
		else:
			# increment the refactory period
			self.refac_time += self.dt
			if(self.refac_time >= self.t_ref):
				# if we've reached the maximum refactory period, reset it
				self.refac = False
				self.refac_time = 0.0
			return 0.0

def lif_ensemble(lifs, encoders):
	lifs = lifs
	encoders = encoders
	neurons = []
	for l_i in range(len(lifs)):
		neurons.append(SpikingLif())

	def spiking_ensemble(x):
		# generate spikes
		spikes = []
		for n_i, neuron in enumerate(neurons):
			spikes.append(
				spiking(
					neuron.spike(
						lifs[n_i](x*encoders[n_i])
					)
				)
			)
		return spikes
	return spiking_ensemble

def z_center(data):
	if(data.size % 2 == 0):
		return np.linspace(-data.size/2, data.size/2, data.size)
	else:
		return np.linspace(-data.size/2, data.size/2, data.size+1)