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

A, neurons = get_activities(rec_lin_neuron, x_vals, n_neurons, x_cepts, max_firing_rates, gain_signs)

fig = plt.figure()
plt.plot(x_vals, A.T)
plt.grid('on')
plt.title("linear rectified neurons")
plt.xlabel("x")
plt.ylabel("Firing Rate (Hz)")
plt.xlim([-1,1])
fig.savefig("1_1a")

decoders, x_hat = get_decoders(A, S, x_vals)
print("decoders %s" %decoders)

plot_xhat(x_vals, x_hat, "noiseless neurons, noiseless decoders", "1_1c")

rmse.append(calc_rmse(x_hat, x_vals))
print("rmse %s" %rmse[-1])

A_noisy = A.T + np.random.normal(scale=0.2*np.max(A), size=A.T.shape)
old_xhat = x_hat
fig = plt.figure()
plt.plot(x_vals, A_noisy)
fig.savefig("noisy_neurons")
new_xhat = np.dot(A_noisy, decoders)

plot_xhat(x_vals, new_xhat, "noisy neurons, noiseless decoders", "1_1d")

rmse.append(calc_rmse(new_xhat, x_vals))
print("rmse %s" %rmse[-1])


decoders_noisy, x_hat_noisy = get_decoders(A_noisy.T, S, x_vals)

plot_xhat(x_vals, x_hat_noisy, "noisy neurons, noisy decoders", "1_1e")

rmse.append(calc_rmse(x_hat_noisy, x_vals))
print("rmse %s" %rmse[-1])

noiseless_xhat = np.dot(A.T, decoders_noisy)

plot_xhat(x_vals, noiseless_xhat, "noiseless neurons, noisy decoders", "1_1e_noiseless")

rmse.append(calc_rmse(noiseless_xhat, x_vals))
print("rmse %s" %rmse[-1])