import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def calc_rmse(predictions, targets):
	return np.sqrt( (np.square(predictions - targets)).mean() )

def drms(sig):
	return np.sqrt(np.sum(np.square(sig))/sig.size)

def z_center(data):
	if(data.size % 2 == 0):
		return np.linspace(-data.size/2, data.size/2, data.size)
	else:
		return np.linspace(-data.size/2, data.size/2, data.size+1)

def ptsc(t, tau, dt=0.001):
	return_val = np.exp(-t/tau) * (t > 0)
	return return_val/(np.sum(return_val)*dt)

# the following two neurons should really be merged into one object

# this neuron returns the firing rate of the lif
def lif_neuron(x_inter, max_fire, t_ref=0.002, t_rc=0.02, radius=2.0):
	beta = 1.0 / (
		1.0 - np.exp(
			(-1.0/max_fire + t_ref) / t_rc
		)
	)
	alpha = (1.0 - beta)/(x_inter + radius*1.0)
	J_bias = 1.0 - alpha * x_inter
	def lif(x):
		J = x * alpha + J_bias
		if(J > 1):
			return 1.0/(t_ref-t_rc*np.log(1-1/J))
		else:
			return 0.0

	return lif

# this neuron returns the current of the lif
def modified_lif(x0_fire, max_fire, t_ref=0.002, t_rc=0.02, radius=2.0):
	beta = 1.0 / (
		1.0 - np.exp(
			(-1.0/max_fire + t_ref) / t_rc
		)
	)
	alpha = (1.0 - beta)/(x0_fire + radius*1.0)
	J_bias = 1.0 - alpha * x0_fire
	def lif_current(x):
		J = x * alpha + J_bias
		if(J > 1):
			return J
		else:
			return 0.0

	return lif_current

def get_activities(neuron_type, x_vals, n_neurons, x_cepts, max_firing_rates, gain_signs, radius=1.0):
	neuron_list = []
	A = np.zeros((n_neurons, x_vals.size))
	for i in range(n_neurons):
		neuron_list.append(neuron_type(x_cepts[i], max_firing_rates[i], radius=radius))
		A[i,:] = neuron_list[i](x_vals*gain_signs[i])
	return A, neuron_list

def get_decoders(A, dx, x_vals):
	gamma = np.dot(A, A.T) * dx
	upsilon = np.dot(A, x_vals) * dx
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

def lif_ensemble(lifs, arg_encoders):
	lifs = lifs
	encoders = arg_encoders
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
						lifs[n_i](
							np.dot(x, encoders[n_i])
						)
					)
				)
			)
		return spikes
	return spiking_ensemble

def lif_ensemble_2d(lifs, arg_encoders):
	lifs = lifs
	encoders = arg_encoders
	neurons = []
	for l_i in range(len(lifs)):
		neurons.append(SpikingLif())

	def spiking_ensemble(x):
		# generate spikes
		spikes = []
		for n_i, neuron in enumerate(neurons):
			#ipdb.set_trace()
			spikes.append(
				spiking(
					neuron.spike(
						lifs[n_i](
							np.dot(x, encoders[n_i])
						)
					)
				)
			)
		return spikes
	return spiking_ensemble

def generate_ensemble(n_neurons, ensemble_type=lif_ensemble, encoders=None):
	max_firing_rates = np.random.uniform(100, 200, n_neurons)
	x_cepts = np.random.uniform(-2, 2, n_neurons)
	if(encoders == None):
		gain_signs = np.random.choice([-1, 1], n_neurons)
	else:
		gain_signs = encoders
	lifs = []
	for i in range(n_neurons):
		lifs.append(
			modified_lif(
				x_cepts[i],
				max_firing_rates[i]
			)
		)
	# okay, this needs to return the lifs...
	return ensemble_type(lifs, gain_signs)

def spike_and_filter(ensemble, input_list, h):
	res = []
	for val in input_list:
		res.append(ensemble(val))
	res = np.array(res)

	# get activities based off of linear function
	A = np.zeros((res.shape), dtype=np.float)
	for i_n in range(res.shape[1]):
		A[:,i_n] = np.convolve(res[:,i_n], h, mode='same')
	return A