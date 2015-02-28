import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import ptsc, get_decoders, generate_ensemble, spike_and_filter

n_neurons = 200

# create the neuron ensembles
dt = 0.001
t_range = np.arange(0, 1, dt)
h_range = np.arange(1000)*dt-0.5
h = ptsc(h_range, 0.005)

# first ensemble
ensemble = generate_ensemble(n_neurons)

# second ensemble
end_ensemble = generate_ensemble(n_neurons)

# this is the goal function to decode
decode_func = lambda x: 2*x + 1

# the first ensemble computes the aformentioned function
x_vals = np.linspace(-2, 2, t_range.size)
A = spike_and_filter(ensemble, x_vals.tolist(), h)

# decode with the modified function (got this from the notes)
first_decoders, x_hat = get_decoders(A.T, A.shape[0], decode_func(x_vals))

# get the decoders for the other ensemble
A = spike_and_filter(end_ensemble, x_vals.tolist(), h)

end_decoders, x_hat = get_decoders(A.T, A.shape[0], x_vals)

# get the activities from the new input
input_func = lambda t: t - 1
A = spike_and_filter(ensemble, input_func(t_range).tolist(), h)

# decode them and plug them into the next population
new_x_hat = np.dot(A, first_decoders)

A = spike_and_filter(end_ensemble, new_x_hat.tolist(), h)

y_hat = np.dot(A, end_decoders)

fig = plt.figure()
plt.plot(t_range, input_func(t_range), label="input")
plt.plot(t_range, decode_func(input_func(t_range)), label="actual")
plt.plot(t_range, y_hat, label="y approximate")
plt.legend()
plt.savefig("4_a")

# test this method on a bunch of other functions
def input_func(t_range, steps=4):
	y_vals = np.random.uniform(-1, 0, steps)
	return_vals = []
	increment = t_range.size/steps
	for t in range(t_range.size):
		return_vals.append(y_vals[t/increment])
	return np.array(return_vals)

input_sig = input_func(t_range)

A = spike_and_filter(ensemble, input_sig.tolist(), h)

new_x_hat = np.dot(A, first_decoders)

A = spike_and_filter(end_ensemble, new_x_hat.tolist(), h)

y_hat = np.dot(A, end_decoders)

fig = plt.figure()
plt.plot(t_range, input_func(t_range), label="input")
plt.plot(t_range, decode_func(input_sig), label="actual")
plt.plot(t_range, y_hat, label="approximate")
plt.legend()
plt.savefig("4_b")

input_func = lambda t: 0.2*np.sin(6*np.pi*t)

A = spike_and_filter(ensemble, input_func(t_range).tolist(), h)

new_x_hat = np.dot(A, first_decoders)

A = spike_and_filter(end_ensemble, new_x_hat.tolist(), h)

y_hat = np.dot(A, end_decoders)

fig = plt.figure()
plt.plot(t_range, input_func(t_range), label="input")
plt.plot(t_range, decode_func(input_func(t_range)), label="actual")
plt.plot(t_range, y_hat, label="approximate")
plt.legend()
plt.savefig("4_c")