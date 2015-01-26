import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import calc_rmse, rec_lin_neuron, lif_neuron, get_activities, get_decoders, plot_xhat

n_neurons = 16
rmse = []

max_firing_rates = np.random.uniform(100, 200, n_neurons)
x_cepts = np.random.uniform(-0.95, 0.95, n_neurons)
gain_signs = np.random.choice([-1, 1], n_neurons)

x_vals = np.arange(-1, 1.05, 0.05)
S = x_vals.size

A, neurons = get_activities(lif_neuron, x_vals, n_neurons, x_cepts, max_firing_rates, gain_signs)

# plot the lif neurons
fig = plt.figure()
plt.plot(x_vals, A.T)
plt.grid('on')
plt.title("leaky-integrate-fire neurons")
plt.xlabel("x")
plt.ylabel("Firing Rate (Hz)")
plt.xlim([-1,1])
fig.savefig("1_3a")

# do noisy generation and decoding
A_noisy = A.T + np.random.normal(scale=0.2*np.max(A), size=A.T.shape)
decoders_noisy, x_hat_noisy = get_decoders(A_noisy.T, S, x_vals)

plot_xhat(x_vals, x_hat_noisy, "noisy neurons, noiseless decoders", "1_3b")

rmse.append(calc_rmse(x_hat_noisy, x_vals))
print("rmse %s" %rmse[-1])

# do noiseless decoding with noisy decoders
new_xhat = np.dot(A.T, decoders_noisy)

plot_xhat(x_vals, new_xhat, "noiseless neurons, noisy decoders", "1_3b_noiseless")

rmse.append(calc_rmse(new_xhat, x_vals))
print("rmse %s" %rmse[-1])