import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, ptsc, generate_ensemble, spike_and_filter, lif_ensemble_2d

def gen_rand_uc_vecs(dims, number):
	vecs = np.random.normal(size=(number,dims))
	print(vecs.shape)
	mags = np.linalg.norm(vecs, axis=-1)
	return vecs / mags[..., np.newaxis]

def get_2d_decoders(A, S, x_vals):
	gamma = np.dot(A, A.T) / S
	# so if I have a two dimensional upsilon....
	upsilon_0 = np.dot(A, x_vals[0]) / S
	upsilon_1 = np.dot(A, x_vals[1]) / S
	# I'll get two dimensional decoders....
	decoders_0 = np.dot(np.linalg.pinv(gamma), upsilon_0)
	decoders_1 = np.dot(np.linalg.pinv(gamma), upsilon_1)
	# Which I can just add up again to get the approximate value
	x_hat = np.dot(A.T, decoders_0) + np.dot(A.T, decoders_1)
	decoders = np.array([decoders_0, decoders_1])
	return decoders, x_hat

def get_ens_dec_2d(n_neurons, x_vals, h, decode_func=None):
	ensemble = generate_ensemble(n_neurons, ensemble_type=lif_ensemble_2d, encoders=gen_rand_uc_vecs(2,n_neurons))
	A = spike_and_filter(ensemble, x_vals.T, h)
	if(decode_func == None):
		decoders = get_2d_decoders(A.T, A.shape[0], decode_func(x_vals))
	else:
		decoders = get_2d_decoders(A.T, A.shape[0], x_vals)
	return ensemble, decoders

# lesson learned, think about higher dimensions before writing code
# the encoder multiplication should have been taken out of the spiking code

n_neurons = 200

dt = 0.001
t_range = np.arange(0, 1, dt)
h_range = np.arange(1000)*dt-0.5
h = ptsc(h_range, 0.005)

a = np.linspace(-1.0,1.0,100)
b = np.linspace(-1.0,1.0,100)
X,Y = np.meshgrid(a, b)
x_vals = [X.reshape((1,-1))[0], Y.reshape((1,-1))[0]]

w_ensemble, w_decoders = get_ens_dec_2d(n_neurons, x_vals, h)
x_ensemble, x_decoders = get_ens_dec_2d(n_neurons, x_vals, h)
y_decode_func = lambda y: -3 * y
y_ensemble, y_decoders = get_ens_dec_2d(n_neurons, x_vals, h, y_decode_func)
q_decode_func = lambda q: -2 * q
q_ensemble, q_decoders = get_ens_dec_2d(n_neurons, x_vals, h, q_decode_func)
z_decode_func = lambda z: 2 * z
z_ensemble, z_decoders = get_ens_dec_2d(n_neurons, x_vals, h, z_decode_func)

# simulate each of them
# this really needs to be fixed
x_input_func = lambda t: np.array([0.5, 1])
A = spike_and_filter(x_ensemble, x_input_func(t_range), h)
x_hat = np.dot(A, x_decoders[0])

fig = plt.figure()
plt.plot(
	x_decode_func(x_input_func(t_range)),
label="x actual")
plt.plot(x_hat, label="x approx")
plt.legend()
plt.savefig("2d_test")

"""
y_input_func = lambda t: np.array([0.1, 0.3])
z_input_func = lambda t: np.array([0.2, 0.1])
q_input_func = lambda t: np.array([0.4, -0.2])

# okay, how do I make an iterator from this exactly?
# either I need iter_shape or op_axes
# I'll use SO for this
A = spike_and_filter(w_ensemble, (x_hat+y_hat+z_hat+q_hat), h)
w_hat = np.dot(A, w_decoders[0]) + np.dot(A, w_decoders[1])

fig = plt.figure()
plt.plot(
	x_decode_func(x_input_func(t_range)) + 
	y_decode_func(y_input_func(t_range)) +
	q_decode_func(q_input_func(t_range)) +
	z_decode_func(z_input_func(t_range)),
label="w actual")
plt.plot(w_hat, label="w approx")
plt.legend()
plt.savefig("5_a")
"""