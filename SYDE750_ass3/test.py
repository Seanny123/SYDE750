# test decoding in one dimension
import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, ptsc, spike_and_filter, lif_ensemble, get_decoders, modified_lif, lif_neuron

def get_ens_dec(n_neurons, x_vals, h, decode_func=None, plot_res=False):
	max_firing_rates = np.random.uniform(100, 200, n_neurons)
	x_cepts = np.random.uniform(-2, 2, n_neurons)
	encoders = np.random.choice([-1, 1], n_neurons)
	
	lif_currents = []
	lif_rates = []
	A = np.zeros( (n_neurons, x_vals.size) )
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
		for i_x, x in enumerate(x_vals):
			A[i,i_x] = lif_rates[i](np.dot(x, encoders[i]))

	ensemble = lif_ensemble(lif_currents, encoders)

	fig = plt.figure()
	plt.plot(x_vals, A.T)
	plt.savefig("decoders")

	A_noisy = A.T + np.random.normal(scale=0.1*np.max(A), size=A.T.shape)
	if(decode_func == None):
		decoders, x_hat = get_decoders(A_noisy.T, 4.0/x_vals.size, x_vals)
		print(decoders)
		if(plot_res):
			print("Plotting")
			fig = plt.figure()
			plt.plot(x_vals.T, label="actual")
			plt.plot(x_hat, label="approx")
			plt.legend()
			#plt.plot(x_hat-x_vals)
			plt.savefig("decode_test")

			fig = plt.figure()
			x_input_func = lambda t: 2*np.sin(3*np.pi*t)
			my_sin = x_input_func(np.linspace(0,1,200))
			A = np.zeros( (n_neurons, x_vals.size) )
			for i in range(n_neurons):
				# get the activities for the decoders
				for i_x, x in enumerate(my_sin):
					A[i,i_x] = lif_rates[i](np.dot(x, encoders[i]))
			plt.plot(my_sin.T, label="actual")
			plt.plot(np.dot(A.T, decoders), label="approx")
			plt.legend()
			plt.savefig("herp")
	else:
		decoders, _ = get_decoders(A_noisy.T, 4.0/x_vals.size, decode_func(x_vals))
	return ensemble, decoders

# lesson learned, think about higher dimensions before writing code
# the encoder multiplication should have been taken out of the spiking code
# what does failing on a constant mean?

n_neurons = 10

dt = 0.001
t_range = np.arange(0, 1, dt)
h_range = np.arange(1000)*dt-0.5
h = ptsc(h_range, 0.005)

x_vals = np.linspace(-2.0,2.0,200)

x_ensemble, x_decoders = get_ens_dec(n_neurons, x_vals, h, decode_func=None, plot_res=True)

# simulate each of them
# something is being scaled. There's only two things that scale anything. h, which we know is normalized and the decoders which we're sure is working correctly
print("simulating")
#x_input_func = lambda t: np.array([0.5]*t.size)
x_input_func = lambda t: 2*np.sin(3*np.pi*t)
A = spike_and_filter(x_ensemble, x_input_func(t_range), h)
x_hat = np.dot(A, x_decoders.T)

fig = plt.figure()
plt.plot(x_input_func(t_range), label="x actual")
plt.plot(x_hat, label="x approx")
plt.legend()
plt.savefig("please")