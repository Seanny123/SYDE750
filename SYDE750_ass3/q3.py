import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, calc_rmse, lif_ensemble, modified_lif, z_center, ptsc, get_decoders

size_list = [8, 16, 32, 64, 128, 256]
dt = 0.001
t_range = np.arange(998)*dt-0.5
h = ptsc(t_range, 0.005)
noise_sig, _ = whitenoise(1, dt, 1, 5, 0)
rmse_list = []

for n_neurons in size_list:
	# create the ensemble
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

	# get the activities
	res_noise = []
	for val in noise_sig:
		res_noise.append(ensemble(val))
	res_noise = np.array(res_noise)

	# filter the spikes
	A = np.zeros((t_range.size, res_noise.shape[1]))
	for i_n in range(len(lifs)):
		A[:,i_n] = np.convolve(res_noise[:,i_n], h, mode='same')

	# create the decoders
	_, x_hat = get_decoders(A.T, A.shape[0], noise_sig)

	# write out the rms # possibly to a file if this takes really long
	rmse_list.append(calc_rmse(x_hat, noise_sig))

fig = plt.figure()
plt.xlabel("number of neurons")
plt.ylabel("rmse_list")
plt.plot(size_list, rmse_list)
plt.title("RMSE decreasing with more neurons")
plt.savefig("3")