import numpy as np
import ipdb
import matplotlib.pyplot as plt

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

from utils import whitenoise, z_center

period = 1

limit_list = [5, 10, 20]
for limit in limit_list:
	res, _ = whitenoise(period, 0.001, 0.5, limit, 0)
	fig = plt.figure()
	plt.plot(res)
	plt.title("Whitenoise Limited to %sHz" %limit)
	plt.xlabel("Time (s)")
	fig.savefig("1_1a_freq%s" %limit)

res, coef = whitenoise(period, 0.001, 0.5, 10, 0)

average_coef = np.zeros(coef.shape)
average_coef = average_coef + np.abs(coef)

for seed in range(1, 99):
	res, coef = whitenoise(period, 0.001, 0.5, 10, seed)
	average_coef = average_coef + np.abs(coef)
average_coef = average_coef / 100
avg = np.fft.fftshift(average_coef)

fig = plt.figure()
plt.plot(z_center(avg), avg)
plt.title("Coefficient Frequencies")
plt.xlabel("$Frequency (Hz)$")
plt.xlim(-50,50)
fig.savefig("1_1b")