import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, ptsc, spike_and_filter, lif_ensemble_2d, modified_lif, lif_neuron

def gen_rand_uc_vecs(dims, number):
	vecs = np.random.normal(size=(number,dims))
	mags = np.linalg.norm(vecs, axis=-1)
	return vecs / mags[..., np.newaxis]

def get_2d_decoders(A, dx, x_vals):
	gamma = np.dot(A, A.T) / S
	# so if I have a two dimensional upsilon....
	upsilon_0 = np.dot(A, x_vals[0]) * dx
	upsilon_1 = np.dot(A, x_vals[1]) * dx
	# I'll get two dimensional decoders....
	decoders_0 = np.dot(np.linalg.pinv(gamma), upsilon_0)
	decoders_1 = np.dot(np.linalg.pinv(gamma), upsilon_1)
	# Which I can just add up again to get the approximate value
	decoders = np.array([decoders_0, decoders_1])
	x_hat = np.dot(A.T, decoders.T)
	return decoders, x_hat

def get_ens_dec_2d(n_neurons, x_vals, h, decode_func=None, plot_res=False):
	max_firing_rates = np.random.uniform(100, 200, n_neurons)
	x_cepts = np.random.uniform(-2, 2, n_neurons)
	encoders = gen_rand_uc_vecs(2,n_neurons)
	
	lif_currents = []
	lif_rates = []
	A = np.zeros( (n_neurons, x_vals[0].size) )
	for i in range(n_neurons):
		lif_currents.append(
			modified_lif(
				x_cepts[i],
				max_firing_rates[i]
			)
		)
		lif_rates.append(
			lif_neuron(
				x_cepts[i],
				max_firing_rates[i]
			)
		)
		# get the activities for the decoders
		for i_x, x in enumerate(x_vals.T):
			A[i,i_x] = lif_rates[i](np.dot(x, encoders[i]))

	ensemble = lif_ensemble_2d(lif_currents, encoders)

	A_noisy = A.T + np.random.normal(scale=0.1*200, size=A.T.shape)
	if(decode_func == None):
		decoders, x_hat = get_2d_decoders(A_noisy.T, 4.0/x_vals[0].size, x_vals)
		if(plot_res):
			print("Plotting")
			fig = plt.figure()
			plt.plot(x_vals.T, label="actual")
			plt.plot(x_hat, label="approx")
			plt.legend()
			plt.savefig("decode_test")
	else:
		decoders, _ = get_2d_decoders(A_noisy.T, 4.0/x_vals[0].size, decode_func(x_vals))
	return ensemble, decoders

# lesson learned, think about higher dimensions before writing code
# the encoder multiplication should have been taken out of the spiking code

n_neurons = 200

dt = 0.001
t_range = np.arange(0, 1, dt)
h_range = np.arange(1000)*dt-0.5
h = ptsc(h_range, 0.005, dt)

a = np.linspace(-2.0,2.0,100)
b = np.linspace(-2.0,2.0,100)
X,Y = np.meshgrid(a, b)
x_vals = np.array([X.reshape((1,-1))[0], Y.reshape((1,-1))[0]])
#x_vals=np.array([a, -b])

w_ensemble, w_decoders = get_ens_dec_2d(n_neurons, x_vals, h, decode_func=None, plot_res=True)
x_ensemble, x_decoders = get_ens_dec_2d(n_neurons, x_vals, h, decode_func=None, plot_res=True)


y_decode_func = lambda y: -3 * y
y_ensemble, y_decoders = get_ens_dec_2d(n_neurons, x_vals, h, y_decode_func)
q_decode_func = lambda q: -2 * q
q_ensemble, q_decoders = get_ens_dec_2d(n_neurons, x_vals, h, q_decode_func)
z_decode_func = lambda z: 2 * z
z_ensemble, z_decoders = get_ens_dec_2d(n_neurons, x_vals, h, z_decode_func)

# simulate each of them
print("simulating")
x_input_func = lambda t: np.array([[0.5, 1],]*t.size)
#x_input_func = lambda t: np.array([np.array([0.5]*t.size), np.sin(3*np.pi*t)]).T
A = spike_and_filter(x_ensemble, x_input_func(t_range), h)
x_hat = np.dot(A, x_decoders.T)

fig = plt.figure()
plt.plot(x_input_func(t_range), label="x actual")
plt.plot(x_hat/10, label="x approx")
plt.legend()
plt.savefig("x_2d_test")

y_input_func = lambda t: np.array([[0.1, 0.3],]*t.size)
A = spike_and_filter(y_ensemble, y_input_func(t_range), h)
y_hat = np.dot(A, y_decoders.T)

fig = plt.figure()
plt.plot(y_decode_func(y_input_func(t_range)), label="y actual")
plt.plot(y_hat, label="y approx")
plt.legend()
plt.savefig("y_2d_test_1")

z_input_func = lambda t: np.array([[0.2, 0.1],]*t.size)
A = spike_and_filter(z_ensemble, z_input_func(t_range), h)
z_hat = np.dot(A, z_decoders.T)

fig = plt.figure()
plt.plot(z_decode_func(z_input_func(t_range)), label="z actual")
plt.plot(z_hat, label="z approx")
plt.legend()
plt.savefig("z_2d_test")

q_input_func = lambda t: np.array([[0.4, -0.2],]*t.size)
A = spike_and_filter(q_ensemble, q_input_func(t_range), h)
q_hat = np.dot(A, q_decoders.T)

fig = plt.figure()
plt.plot(q_decode_func(q_input_func(t_range)), label="q actual")
plt.plot(q_hat, label="q approx")
plt.legend()
plt.savefig("q_2d_test_1")

A = spike_and_filter(w_ensemble, (x_hat+y_hat+z_hat+q_hat), h)
w_hat = np.dot(A, w_decoders.T)

fig = plt.figure()
plt.plot(
	x_input_func(t_range) + 
	y_decode_func(y_input_func(t_range)) +
	q_decode_func(q_input_func(t_range)) +
	z_decode_func(z_input_func(t_range)),
label="w actual")
plt.plot(w_hat, label="w approx")
plt.legend()
plt.savefig("6_a")

y_input_func = lambda t: np.array([np.sin(4*np.pi*t), np.array([0.3]*t.size)]).T
A = spike_and_filter(y_ensemble, y_input_func(t_range), h)
y_hat = np.dot(A, y_decoders.T)

fig = plt.figure()
plt.plot(y_decode_func(y_input_func(t_range)), label="y actual")
plt.plot(y_hat, label="y approx")
plt.legend()
plt.savefig("y_2d_test_2")

q_input_func = lambda t: np.array([np.sin(4*np.pi*t), np.array([-0.2]*t.size)]).T
A = spike_and_filter(q_ensemble, q_input_func(t_range), h)
q_hat = np.dot(A, q_decoders.T)

fig = plt.figure()
plt.plot(q_decode_func(q_input_func(t_range)), label="q actual")
plt.plot(q_hat, label="q approx")
plt.legend()
plt.savefig("q_2d_test_2")

A = spike_and_filter(w_ensemble, (x_hat+y_hat+z_hat+q_hat), h)
w_hat = np.dot(A, w_decoders.T)

fig = plt.figure()
plt.plot(
	x_input_func(t_range) + 
	y_decode_func(y_input_func(t_range)) +
	q_decode_func(q_input_func(t_range)) +
	z_decode_func(z_input_func(t_range)),
label="w actual")
plt.plot(w_hat, label="w approx")
plt.legend()
plt.savefig("6_b")