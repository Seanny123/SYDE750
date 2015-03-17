import nengo
from nengo.dists import Uniform
from nengo.utils.ensemble import tuning_curves
import matplotlib.pyplot as plt
import numpy as np
import ipdb

def calc_rmse(x, xhat):
	return np.sqrt(np.average((x-xhat)**2))

model = nengo.Network()

with model:
	ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002))
	connection = nengo.Connection(ens, ens)

sim = nengo.Simulator(model)
x_vals, A = tuning_curves(ens, sim)

# plot the tuning tuning curves
fig = plt.figure()
plt.plot(x_vals, A)
plt.title("Tuning Curves")
fig.savefig("1_a_1")

# plot the representation
d = sim.data[connection].decoders.T
xhat = np.dot(A, d)
fig = plt.figure()
plt.plot(x_vals, x_vals, label="$x$")
plt.plot(x_vals, xhat, label= "$\hat{x}$")
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.legend()
fig.savefig("1_a_2")

# note the rmse
print "RMSE %s" %calc_rmse(x_vals, xhat)

rmse_list = []
radius_list = [1.0, 2.0, 3.0, 4.0]
for radius in radius_list:
	model = nengo.Network()

	with model:
		ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002), radius=radius)
		connection = nengo.Connection(ens, ens)

	sim = nengo.Simulator(model)
	x_vals, A = tuning_curves(ens, sim)

	d = sim.data[connection].decoders.T
	xhat = np.dot(A, d)

	rmse_list.append(calc_rmse(x_vals, xhat))

fig = plt.figure()
plt.title("RMSE from increasing radius")
plt.plot(radius_list, rmse_list)
fig.savefig("1_b")

rmse_list = []
tau_ref_list = [0.002, 0.003, 0.004, 0.005]
for tau_ref in tau_ref_list:
	model = nengo.Network()

	with model:
		ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=tau_ref))
		connection = nengo.Connection(ens, ens)

	sim = nengo.Simulator(model)
	x_vals, A = tuning_curves(ens, sim)

	d = sim.data[connection].decoders.T
	xhat = np.dot(A, d)

	rmse_list.append(calc_rmse(x_vals, xhat))

fig = plt.figure()
plt.title("RMSE from increasing tau_ref")
plt.plot(tau_ref_list, rmse_list)
fig.savefig("1_c")

rmse_list = []
tau_rc_list = [0.02, 0.04, 0.08, 0.16]
for tau_rc in tau_rc_list:
	model = nengo.Network()

	with model:
		ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIF(tau_rc=tau_rc, tau_ref=0.002))
		connection = nengo.Connection(ens, ens)

	sim = nengo.Simulator(model)
	x_vals, A = tuning_curves(ens, sim)

	d = sim.data[connection].decoders.T
	xhat = np.dot(A, d)

	rmse_list.append(calc_rmse(x_vals, xhat))

fig = plt.figure()
plt.title("RMSE from increasing tau_rc")
plt.plot(tau_rc_list, rmse_list)
fig.savefig("1_d")