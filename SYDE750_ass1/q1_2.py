import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import calc_rmse, rec_lin_neuron, lif_neuron, get_activities, get_decoders, plot_xhat

def q1_2(noise_val, filename):
	N_vals = [4, 8, 16, 32, 64, 128, 256]#, 512, 1024, 2048]#, 4096, 8192]
	err_dist = []
	err_noise = []
	x_vals = np.arange(-1, 1.05, 0.05)

	for n_neurons in N_vals:
		print("N_val: %s" %n_neurons)
		e_dist_avg = 0.0
		e_noise_avg = 0.0

		for run_num in range(5):
			max_firing_rates = np.random.uniform(100, 200, n_neurons)
			x_cepts = np.random.uniform(-0.95, 0.95, n_neurons)
			gain_signs = np.random.choice([-1, 1], n_neurons)

			# generate without noise
			A, neurons = get_activities(rec_lin_neuron, x_vals, n_neurons, x_cepts, max_firing_rates, gain_signs)
			S = x_vals.size
			decoders, x_hat = get_decoders(A, S, x_vals)
			# get E_dist
			e_dist = calc_rmse(x_hat, x_vals)
			e_dist_avg += e_dist

			# generate with noise
			A_noisy = A.T + np.random.normal(scale=0.2*np.max(A), size=A.T.shape)
			decoders_noisy, x_hat_noisy = get_decoders(A_noisy.T, S, x_vals)
			# get E_noise
			e_noise = calc_rmse(x_hat_noisy, x_vals)
			e_noise_avg += (e_noise - e_dist)

		err_dist.append(e_dist_avg/5)
		err_noise.append(e_noise_avg/5)

	# plot the result
	over_N = lambda N: 1.0/N
	over_N_squared = lambda N: 1.0/np.square(N)
	over_N_4 = lambda N: 1.0/np.power(N, 4)
	
	# plot the distortion proportional to 1/N^2
	fig = plt.figure()
	plt.loglog(N_vals, over_N_squared(np.array(N_vals)), label="1/N^2")
	plt.loglog(N_vals, over_N_4(np.array(N_vals)), label="1/N^4")
	plt.loglog(N_vals, err_dist, label="Distortion Error")
	#plt.set_yscale('log')
	#plt.set_xscale('log')
	plt.ylabel("Error")
	plt.xlabel("Number of Neurons")
	plt.title("Distortion Error with Noise=%s" %noise_val)
	plt.legend(loc=3)
	fig.savefig("dist_%s" %filename)

	# plot the noise error proportional to 1/N
	fig = plt.figure()
	plt.loglog(N_vals, over_N(np.array(N_vals)), label="1/N")
	plt.loglog(N_vals, err_dist, label="Noise Error")
	#plt.set_yscale('log')
	#plt.set_xscale('log')
	plt.ylabel("Error")
	plt.xlabel("Number of Neurons")
	plt.title("Noise Error with Noise=%s" %noise_val)
	plt.legend(loc=3)
	fig.savefig("noise_%s" %filename)

q1_2(0.1, "1_2_high")
q1_2(0.01, "1_2_low")