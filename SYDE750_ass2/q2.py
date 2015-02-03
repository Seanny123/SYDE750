def modified_lif(x0_fire, max_fire, t_ref=0.002, t_rc=0.02):
	beta = 1.0 / (
		1.0 - np.exp(
			(-1.0/max_fire + t_ref) / t_rc
		)
	)
	J_bias = 1.0 / (
		1.0 - np.exp(
			(-1.0/x0_fire + t_ref) / t_rc
		)
	)
	alpha = beta - J_bias
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

lif_neuron = modified_lif(40, 150)
