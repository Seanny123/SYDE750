import numpy as np
import matplotlib.pyplot as plt

def whitenoise(T, dt, rms, limit, seed):
	# randomly generate co-efficient from a gaussian distribution equal to half the size of frequencies
	np.random.seed(seed)
	coef = np.random.normal(0, 1,
					int( np.ceil(T/(2.0*dt)) )
				) + 1j * np.random.normal(0, 1,
					int( np.ceil(T/(2.0*dt)) )
				)

	# eliminate anything over the limit
	# wait, why does this work?
	frequencies = np.arange(0, coef.size*2, 2.0/T)
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

	#frequencies = np.fft.fftfreq(final_coef.size, d=dt)
	#final_coef[frequencies > limit] = 0.0

	# bring it to the time domain
	time_domain = np.fft.ifft(final_coef)
	# make it totally real
	time_domain = np.real(time_domain)
	# calculate the rms and scale it
	curr_rms = np.sqrt(np.sum(np.square(time_domain))/T)
	time_domain = (rms/curr_rms)*time_domain
	return time_domain, (rms/curr_rms)*final_coef

class SpikingLif(object):
	def __init__(self, t_ref=0.002, t_rc=0.2, dt=0.0001):
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
			dV = 1/self.t_rc*(current-self.potential)*self.dt
			self.potential += dV
			if(self.potential >= 1):
				self.refac = True
				self.potential = 0.0
				self.spike_count += 1
				return 2.0
			return self.potential
		else:
			# increment the refactory period
			self.refac_time += self.dt
			if(self.refac_time >= self.t_ref):
				self.refac = False
				self.refac_time = 0.0
			return 0.0

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

def two_neurons():
	lif1 = modified_lif(40, 150)
	lif2 = modified_lif(40, 150)
	n1 = SpikingLif()
	n2 = SpikingLif()

	def two_spikes(x):
		spike1 = (n1.spike(
			lif1(np.array([x]))
		) < 2)
		spike2 = (n2.spike(
			lif2(np.array([x*-1]))
		) < 2)
		return spike1, spike2
	return two_spikes