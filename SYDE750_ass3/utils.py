def calc_rmse(predictions, targets):
	return np.sqrt( (np.square(predictions - targets)).mean() )

def lif_neuron(x_inter, max_fire, t_ref=0.002, t_rc=0.02):
	beta = 1.0 / (
		1.0 - np.exp(
			(-1.0/max_fire + t_ref) / t_rc
		)
	)
	alpha = (1.0 - beta)/(x_inter + 1.0)
	J_bias = 1.0 - alpha * x_inter
	def lif(x):
		J = x * alpha + J_bias
		return_val = np.zeros(x.shape[0])
		# Select all the values where J > 1
		return_val[J > 1] += np.maximum(
						# Caluclate the activity
						1.0/(t_ref-t_rc*np.log(1-1/J[J > 1])),
						# make it zero if it's below zero
						np.zeros(return_val[J > 1].size)
					)
		return return_val
	return lif

def get_activities(neuron_type, x_vals, n_neurons, x_cepts, max_firing_rates, gain_signs):
	neuron_list = []
	A = np.zeros((n_neurons, x_vals.size))
	for i in range(n_neurons):
		neuron_list.append(neuron_type(x_cepts[i], max_firing_rates[i]))
		A[i,:] = neuron_list[i](x_vals*gain_signs[i])
	return A, neuron_list

def get_decoders(A, S, x_vals):
	gamma = np.dot(A, A.T) / S
	upsilon = np.dot(A, x_vals) / S
	decoders = np.dot(np.linalg.pinv(gamma), upsilon)
	ipdb.set_trace()
	x_hat = np.dot(A.T, decoders)
	return decoders, x_hat

def plot_xhat(x_vals, x_hat, title, filename):
	# plot x_hat overlaid with x
	fig = plt.figure()
	plt.plot(x_vals, x_vals, label="real value")
	plt.plot(x_vals, x_hat, label="approximated value")
	plt.title("x_hat %s" %title)
	plt.xlabel("x")
	plt.ylabel("Firing Rate (Hz)")
	plt.xlim([-1,1])
	plt.legend(loc=4)
	plt.savefig("%s_1" %filename)

	# plot x_hat-x
	fig = plt.figure()
	plt.plot(x_vals, (x_vals - x_hat))
	plt.title("approximation error %s" %title)
	plt.xlabel("x")
	plt.ylabel("Error")
	plt.xlim([-1,1])
	plt.savefig("%s_2" %filename)