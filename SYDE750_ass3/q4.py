import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, calc_rmse, lif_ensemble, modified_lif, ptsc, get_decoders

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

n_neurons = 200

# create the neuron ensembles
dt = 0.001
t_range = np.arange(0, 1, dt)
h_range = np.arange(1000)*dt-0.5
h = ptsc(h_range, 0.005)

max_firing_rates = np.random.uniform(100, 200, n_neurons)
x_cepts = np.random.uniform(-2, 2, n_neurons)
gain_signs = np.random.choice([-1, 1], n_neurons)
lifs = []
for i in range(n_neurons):
	lifs.append(
		modified_lif(
			x_cepts[i],
			max_firing_rates[i]
		)
	)
ensemble = lif_ensemble(lifs, gain_signs)

# this is the goal function to decode
decode_func = lambda x: 2*x + 1

# the first ensemble computes the aformentioned function
x_vals = np.linspace(-2, 2, t_range.size)
res = []
for val in np.nditer(x_vals):
	res.append(ensemble(val))
res = np.array(res)

# get activities based off of linear function
A = np.zeros((t_range.size, res.shape[1]))
for i_n in range(len(lifs)):
	A[:,i_n] = np.convolve(res[:,i_n], h, mode='same')

# decode with the modified function (got this from the notes)
first_decoders, x_hat = get_decoders(A.T, A.shape[0], decode_func(x_vals))

"""
# check that it works okay
fig = plt.figure()
plt.plot(t_range, decode_func(x_vals), label="input")
plt.plot(t_range, x_hat, label="approximate")
plt.legend()
plt.savefig("funky_decoder")

# get the activities from the new input
input_func = lambda t: t - 1
res = []
for val in np.nditer(input_func(t_range)):
	res.append(ensemble(val))
res = np.array(res)

A = np.zeros((t_range.size, res.shape[1]))
for i_n in range(len(lifs)):
	A[:,i_n] = np.convolve(res[:,i_n], h, mode='same')

# decode them to make sure they aren't totally absurd
# OH GOD. THEY'RE TERRIBLE. WHAT HAPPENED. ARE THEY SUPPOSED TO BE THIS BAD?
new_x_hat = np.dot(A, first_decoders)
fig = plt.figure()
plt.plot(t_range, input_func(t_range), label="input")
plt.plot(t_range, decode_func(input_func(t_range)), label="actual")
plt.plot(t_range, new_x_hat, label="approximate")
plt.legend()
plt.savefig("4_a")
"""

# test this method on a bunch of other functions
def input_func(t_range, steps=2):
	y_vals = np.random.uniform(-1, 0, steps)
	return_vals = []
	increment = t_range.size/steps
	for t in range(t_range.size):
		return_vals.append(y_vals[t/increment])
	return np.array(return_vals)

res = []
for val in np.nditer(input_func(t_range)):
	res.append(ensemble(val))
res = np.array(res)

A = np.zeros((t_range.size, res.shape[1]))
for i_n in range(len(lifs)):
	A[:,i_n] = np.convolve(res[:,i_n], h, mode='same')

# decode them to make sure they aren't totally absurd
new_x_hat = np.dot(A, first_decoders)

fig = plt.figure()
plt.plot(t_range, input_func(t_range), label="input")
plt.plot(t_range, decode_func(input_func(t_range)), label="actual")
plt.plot(t_range, new_x_hat, label="approximate")
plt.legend()
plt.savefig("4_b")

"""
input_func = lambda t: 0.2*np.sin(6*np.pi*t)

res = []
for val in np.nditer(input_func(t_range)):
	res.append(ensemble(val))
res = np.array(res)

A = np.zeros((t_range.size, res.shape[1]))
for i_n in range(len(lifs)):
	A[:,i_n] = np.convolve(res[:,i_n], h, mode='same')

# decode them to make sure they aren't totally absurd
new_x_hat = np.dot(A, first_decoders)

fig = plt.figure()
plt.plot(t_range, input_func(t_range), label="input")
plt.plot(t_range, decode_func(input_func(t_range)), label="actual")
plt.plot(t_range, new_x_hat, label="approximate")
plt.legend()
plt.savefig("4_c")
"""