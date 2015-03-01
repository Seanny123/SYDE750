import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, ptsc, get_decoders, generate_ensemble, spike_and_filter

# not used, because a little excessive in terms of refactoring
def get_ens_dec(n_neurons, x_vals, h, decode_func=None):
	ensemble = generate_ensemble(n_neurons)
	A = spike_and_filter(x_ensemble, x_vals, h)
	decoders = get_decoders(A.T, A.shape[0], x_decode_func(x_vals))
	return ensemble, decoders

n_neurons = 200

# create the neuron ensembles
dt = 0.001
t_range = np.arange(0, 1, dt)
h_range = np.arange(1000)*dt-0.5
h = ptsc(h_range, 0.005)

x_vals = np.linspace(-2, 2, t_range.size)

x_ensemble = generate_ensemble(n_neurons)
A = spike_and_filter(x_ensemble, x_vals, h)
x_decode_func = lambda x: 0.5*x
x_decoders, x_hat = get_decoders(A.T, A.shape[0], x_decode_func(x_vals))

y_ensemble = generate_ensemble(n_neurons)
A = spike_and_filter(y_ensemble, x_vals, h)
y_decode_func = lambda y: 2*y
y_decoders, _ = get_decoders(A.T, A.shape[0], y_decode_func(x_vals))

z_ensemble = generate_ensemble(n_neurons)
A = spike_and_filter(z_ensemble, x_vals, h)
z_decoders, _ = get_decoders(A.T, A.shape[0], x_vals)


x_input_func = lambda t: np.cos(3*np.pi*t)
A = spike_and_filter(x_ensemble, x_input_func(t_range), h)
x_hat = np.dot(A, x_decoders)

y_input_func = lambda t: 0.5*np.sin(2*np.pi*t)
A = spike_and_filter(y_ensemble, y_input_func(t_range), h)
y_hat = np.dot(A, y_decoders)

A = spike_and_filter(z_ensemble, (x_hat+y_hat), h)
z_hat = np.dot(A, z_decoders)

fig = plt.figure()
plt.plot(x_decode_func(x_input_func(t_range)), label="x input")
plt.plot(y_decode_func(y_input_func(t_range)), label="y input")
#plt.plot(x_hat, label="x approx")
#plt.plot(y_hat, label="y approx")
plt.plot(x_decode_func(x_input_func(t_range))+y_decode_func(y_input_func(t_range)), label="z actual")
plt.plot(z_hat, label="z approx")
plt.legend()
plt.savefig("5_a")

h_range = np.arange(998)*dt-0.5
h = ptsc(h_range, 0.005)

x_noise, _ = whitenoise(1, dt, 1, 8, 0)
A = spike_and_filter(x_ensemble, x_noise, h)
x_hat = np.dot(A, x_decoders)

y_noise, _ = whitenoise(1, dt, 0.5, 5, 0)
A = spike_and_filter(y_ensemble, y_noise, h)
y_hat = np.dot(A, y_decoders)

A = spike_and_filter(z_ensemble, (x_hat+y_hat), h)
z_hat = np.dot(A, z_decoders)

fig = plt.figure()
plt.plot(x_noise, label="x input")
plt.plot(y_noise, label="y input")
plt.plot(x_noise+y_noise, label="z actual")
plt.plot(z_hat, label="z approx")
plt.legend()
plt.savefig("5_b")