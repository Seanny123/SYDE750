import numpy as np
import ipdb
import matplotlib.pyplot as plt

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def failed_whitenoise(T, dt, rms, limit, seed):
	# randomly generate co-efficient from a gaussian distribution equal to half the size of frequencies
	np.random.seed(seed)
	# Why does the period result from the size of my coefficients?
	# 1/dt / 1/T
	coef = np.random.normal(0, 1,
					int( np.ceil(T/(2.0*dt)) )
				) + 1j * np.random.normal(0, 1,
					int( np.ceil(T/(2.0*dt)) )
				)

	# eliminate anything over the limit
	frequencies = np.arange(0, coef.size, 1/T)
	coef[frequencies > limit] = 0.0
	coef[0] = 0.0
	if(coef.size % 2 == 1):
		print("odd")
		final_coef = np.zeros(coef[1:].size * 2 + 1, dtype=np.complex_)
		final_coef[coef.size] = coef
		final_coef[coef.size:] = coef[1:][::-1].conj()
	else:
		print("even")
		final_coef = np.zeros(coef[1:-1].size * 2 + 2, dtype=np.complex_)
		# Don't touch the DC or the middle term!
		final_coef[:coef.size] = coef
		final_coef[coef.size:] = coef[1:-1][::-1].conj()

	# bring it to the time domain
	time_domain = np.fft.ifft(final_coef)
	# make it totally real
	time_domain = np.real(time_domain)
	# calculate the rms and scale it
	curr_rms = np.sqrt(np.sum(np.square(time_domain))/T)
	time_domain = (rms/curr_rms)*time_domain
	return time_domain

res = failed_whitenoise(1, 0.001, 0.5, 5, 0)
#plt.figure()
#plt.plot(res)
#plt.plot(np.imag(res))
#plt.show()
#ipdb.set_trace()