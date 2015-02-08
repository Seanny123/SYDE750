import numpy as np
import ipdb
import matplotlib.pyplot as plt

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

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
	frequencies = np.arange(0, coef.size*2/T, 2.0/T)
	coef_mult = []
	for freq in np.nditer(frequencies):
		#ipdb.set_trace()
		scale = np.exp(-freq**2/(2*bandwidth**2))
		if(scale <= 0.0):
			coef_mult.append(0.0)
		else:
			coef_mult.append(
				np.random.normal(0, scale)
			)
	coef_mult = np.array(coef_mult)
	coef = coef * coef_mult
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

period = 1

limit_list = [5, 10, 20]
for limit in limit_list:
	res, _ = gps_whitenoise(period, 0.001, 0.5, limit, 0)
	fig = plt.figure()
	plt.plot(res)
	plt.title("Whitenoise Limited to %sHz" %limit)
	plt.xlabel("Time (s)")
	fig.savefig("1_2a_freq%s" %limit)

res, coef = gps_whitenoise(period, 0.001, 0.5, 10, 0)

average_coef = np.zeros(coef.shape)
average_coef = average_coef + np.abs(coef)

for seed in range(1, 99):
	res, coef = gps_whitenoise(period, 0.001, 0.5, 10, seed)
	average_coef = average_coef + np.abs(coef)
average_coef = average_coef / 100
avg = np.fft.fftshift(average_coef)

omega = np.linspace(-1.0/0.001, 1.0/0.001, average_coef.size)
fig = plt.figure()
plt.plot(omega, avg)
plt.title("Coefficient Frequencies")
plt.xlabel("$\omega$")
plt.xlim(-50,50)
fig.savefig("1_2b")