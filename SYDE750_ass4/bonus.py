# A double integrator, as seen in the nengo User Guide
# https://pythonhosted.org/nengo/examples/network_design.html

import nengo
import numpy as np
from nengo.dists import Uniform
from nengo.utils.functions import piecewise
import matplotlib.pyplot as plt
import ipdb

model = nengo.Network()

with model:
	my_input = nengo.Node(piecewise({0: 0, 0.2: 0.5, 1: 0, 2: -1, 3: 0, 4: 1, 5: 0}))

	# First integrator
	pre_ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIFRate(tau_rc=0.02, tau_ref=0.002))
	connection = nengo.Connection(pre_ens, pre_ens, synapse=0.1)

	# Second integrator
	post_ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIFRate(tau_rc=0.02, tau_ref=0.002))
	connection = nengo.Connection(post_ens, post_ens, synapse=0.1)
	nengo.Connection(pre_ens, post_ens, synapse=0.05)

	# Connect the two
	nengo.Connection(my_input, pre_ens, synapse=None)
	nengo.Connection(pre_ens, post_ens, synapse=None, transform=0.1)

	input_probe = nengo.Probe(my_input)
	pre_probe = nengo.Probe(pre_ens, synapse=0.01)
	post_probe = nengo.Probe(post_ens, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(5.0)

fig = plt.figure()
plt.plot(sim.trange(), sim.data[input_probe], label="input")
plt.plot(sim.trange(), sim.data[pre_probe], label="pre ensemble")
plt.plot(sim.trange(), sim.data[post_probe], label="post ensemble")
plt.legend()
fig.savefig("3_f")