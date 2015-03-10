import nengo
from nengo.dists import Uniform
from nengo.utils.ensemble import tuning_curves
import matplotlib.pyplot
import numpy as np
import ipdb

model = nengo.Network()

with model:
	ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002))
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
xhat = numpy.dot(A, d)
fig = plt.figure()
plt.plot(x_vals, x_vals, label="$x$")
plt.plot(x_vals, xhat, label= "$\hat{x}$")
ylim(-1, 1)
xlim(-1, 1)
plt.legend()
fig.savefig("1_a_2")

# note the rmse
print 'RMSE', np.sqrt(np.average((x-xhat)**2))