import numpy as np
import ipdb
import matplotlib.pyplot as plt

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

from utils import whitenoise

limit_list = [5, 10, 20]
for limit in limit_list:
	res, _ = whitenoise(1, 0.001, 0.5, limit, 0)
	fig = plt.figure()
	plt.plot(res)
	plt.title("Whitenoise Limited to %sHz" %limit)
	plt.xlabel("Time (s)")
	fig.savefig("1_1a_freq%s" %limit)

res, coef = whitenoise(1, 0.001, 0.5, 10, 0)

average_coef = np.zeros(coef.shape)
average_coef = average_coef + np.abs(coef)

for _ in range(99):
	res, coef = whitenoise(1, 0.001, 0.5, 10, 0)
	average_coef = average_coef + np.abs(coef)
average_coef = average_coef / 100
avg = np.fft.fftshift(average_coef)

omega = np.linspace(-1.0/0.001, 1.0/0.001, average_coef.size)
fig = plt.figure()
plt.plot(omega, avg)
plt.title("Coefficient Frequencies")
plt.xlabel("$\omega$")
plt.xlim(-50,50)
fig.savefig("1_1b")