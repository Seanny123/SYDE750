import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import calc_rmse, lif_neuron, get_activities, get_decoders, plot_xhat

# Generate some neurons and plot the tuning curves
n_neurons = 16
rmse = []

max_firing_rates = np.random.uniform(100, 200, n_neurons)
x_cepts = np.random.uniform(-2, 2, n_neurons)
encoders = np.random.choice([-1, 1], n_neurons)

x_vals = np.arange(-2.55, 2.55, 0.05)
S = x_vals.size

lifs = []
A = np.zeros( (n_neurons, x_vals.size) )
for i in range(n_neurons):
	lifs.append(
		lif_neuron(
			x_cepts[i],
			max_firing_rates[i]
		)
	)
	# get the activities for the decoders
	for i_x, x in enumerate(x_vals):
		A[i,i_x] = lifs[i](np.dot(x, encoders[i]))

# plot the lif neurons
fig = plt.figure()
plt.plot(x_vals, A.T)
plt.grid('on')
plt.title("leaky-integrate-fire neurons")
plt.xlabel("x")
plt.ylabel("Firing Rate (Hz)")
plt.xlim([-2.5,2.5])
fig.savefig("1_1a")

# do noisy generation and decoding
A_noisy = A.T + np.random.normal(scale=0.1*200, size=A.T.shape)
decoders_noisy, x_hat_noisy = get_decoders(A_noisy.T, S, x_vals)

plot_xhat(x_vals, x_hat_noisy, "noisy neurons, noisy decoders", "1_1b")