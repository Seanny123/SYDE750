import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, z_center

# what's that about my axises? I need to multiply this stuff by 2*pi?

T_list = [1.0, 2.0, 4.0]
limit = 20

for i_p, period in enumerate(T_list):
	res, coef = whitenoise(period, 0.001, rms=0.5, limit=20, seed=3)
	fig = plt.figure()
	plt.plot(res)
	plt.title("Period %s" %period)
	plt.xlabel("Time (s)")
	fig.savefig("check_period_%s" %i_p)

for i_p, period in enumerate(T_list):
	res, coef = whitenoise(period, 0.001, rms=0.5, limit=limit, seed=1)

	average_coef = np.zeros(coef.shape)
	average_coef = average_coef + np.abs(coef)

	for seed in range(1, 99):
		res, coef = whitenoise(period, 0.001, rms=0.5, limit=limit, seed=seed)
		average_coef = average_coef + np.abs(coef)
	average_coef = average_coef / 100
	avg = np.fft.fftshift(average_coef)

	fig = plt.figure()
	plt.plot(z_center(avg), avg)
	plt.title("Coefficient Frequencies")
	plt.xlabel("$\omega$")
	plt.xlim(-50,50)
	fig.savefig("check_T_freq_%s" %i_p)