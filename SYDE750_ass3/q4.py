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

# first ensemble
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

# second ensemble
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
end_ensemble = lif_ensemble(lifs, gain_signs)

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

# get the decoders for the other thing... huh... how do I do that...
x_vals = np.linspace(-2, 2, t_range.size)
res = []
for val in np.nditer(x_vals):
	res.append(end_ensemble(val))
res = np.array(res)

# get activities based off of linear function
A = np.zeros((t_range.size, res.shape[1]))
for i_n in range(len(lifs)):
	A[:,i_n] = np.convolve(res[:,i_n], h, mode='same')

end_decoders, x_hat = get_decoders(A.T, A.shape[0], x_vals)

# get the activities from the new input
input_func = lambda t: t - 1
res = []
for val in np.nditer(input_func(t_range)):
	res.append(ensemble(val))
res = np.array(res)

A = np.zeros((t_range.size, res.shape[1]))
for i_n in range(len(lifs)):
	A[:,i_n] = np.convolve(res[:,i_n], h, mode='same')

# decode them and plug them into the next population
new_x_hat = np.dot(A, first_decoders)

res = []
for val in np.nditer(new_x_hat):
	res.append(end_ensemble(val))
res = np.array(res)

A = np.zeros((t_range.size, res.shape[1]))
for i_n in range(len(lifs)):
	A[:,i_n] = np.convolve(res[:,i_n], h, mode='same')

y_hat = np.dot(A, end_decoders)

fig = plt.figure()
plt.plot(t_range, input_func(t_range), label="input")
plt.plot(t_range, decode_func(input_func(t_range)), label="actual")
plt.plot(t_range, new_x_hat, label="x approximate")
plt.plot(t_range, y_hat, label="y approximate")
plt.legend()
plt.savefig("4_a")

"""
# test this method on a bunch of other functions
def input_func(t_range, steps=4):
	y_vals = np.random.uniform(-1, 0, steps)
	return_vals = []
	increment = t_range.size/steps
	for t in range(t_range.size):
		return_vals.append(y_vals[t/increment])
	return np.array(return_vals)

input_sig = input_func(t_range)

res = []
for val in np.nditer(input_sig):
	res.append(ensemble(val))
res = np.array(res)

A = np.zeros((t_range.size, res.shape[1]))
for i_n in range(len(lifs)):
	A[:,i_n] = np.convolve(res[:,i_n], h, mode='same')

# decode them to make sure they aren't totally absurd
# why can't it just track the value -0.5?
new_x_hat = np.dot(A, first_decoders)

fig = plt.figure()
plt.plot(t_range, input_func(t_range), label="input")
plt.plot(t_range, decode_func(input_sig), label="actual")
plt.plot(t_range, new_x_hat, label="approximate")
plt.legend()
plt.savefig("4_b")

noise_sig, _ = whitenoise(1, dt, 1, 5, 0)

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