import numpy as np
import ipdb
import matplotlib.pyplot as plt

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

# just pass in f_x into the decoders of the first population
# do I mean the activity matrix?
# do I just get the activities from the previous deocder matrix?

n_neurons = 200

# regenerate the decoders in the old-fashioned way for both populations
dt = 0.001
t_range = np.arange(0, 1, dt)


input_func = lambda t: t - 1

fig = plt.figure()
plt.plot(t_range, input_func(t_range))
plt.savefig("4_a")

def input_func(t_range):
	y_vals = np.random.uniform(-1, 0, 10)
	return_vals = []
	increment = t_range.size/10
	for t in range(t_range.size):
		return_vals.append(y_vals[t/increment])
	return np.array(return_vals)

fig = plt.figure()
plt.plot(t_range, input_func(t_range))
plt.savefig("4_b")

input_func = lambda t: 0.2*np.sin(6*np.pi*t)

fig = plt.figure()
plt.plot(t_range, input_func(t_range))
plt.savefig("4_c")