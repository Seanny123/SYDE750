import numpy as np
import ipdb
import matplotlib.pyplot as plt

# change the filter from a box to a gaussian
def gps_whitenoise(T, dt, rms, bandwidth, seed):
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
	# so I understand that I have to multiply something here, but what the fuck is that
	# like, do I sample a bunch of values for each frequency and then multiply it with my existing frequencies?
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