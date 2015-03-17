import nengo
import numpy as np
from nengo.dists import Uniform
from nengo.utils.functions import piecewise
import matplotlib.pyplot as plt
import ipdb

model = nengo.Network()

with model:
	my_input = nengo.Node(piecewise({0:0, 0.04:0.9, 1.0:0}))
	ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002))
	nengo.Connection(my_input, ens, synapse=0.005)
	connection = nengo.Connection(ens, ens, synapse=0.05)

	input_probe = nengo.Probe(my_input)
	ens_probe = nengo.Probe(ens)

sim = nengo.Simulator(model)
sim.run(1.5)

fig = plt.figure()
plt.plot(sim.trange(), sim.data[input_probe], label="input")
plt.plot(sim.trange(), sim.data[ens_probe], label="ensemble")
plt.legend()
fig.savefig("3_a")


model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
sim = nengo.Simulator(model)
sim.run(1.5)

fig = plt.figure()
plt.plot(sim.trange(), sim.data[input_probe], label="input")
plt.plot(sim.trange(), sim.data[ens_probe], label="ensemble")
plt.legend()
fig.savefig("3_b")


model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
model.nodes[0].output = piecewise({0:0, 0.04:0.9, 0.16:0})
sim = nengo.Simulator(model)
sim.run(1.5)

fig = plt.figure()
plt.plot(sim.trange(), sim.data[input_probe], label="input")
plt.plot(sim.trange(), sim.data[ens_probe], label="ensemble")
plt.legend()
fig.savefig("3_c")


model.nodes[0].output = piecewise({0:lambda t: 2*t, 0.45:0})
sim = nengo.Simulator(model)
sim.run(1.5)

fig = plt.figure()
plt.plot(sim.trange(), sim.data[input_probe], label="input")
plt.plot(sim.trange(), sim.data[ens_probe], label="ensemble")
plt.legend()
fig.savefig("3_d")

model.nodes[0].output = lambda t: 5*np.sin(5*t)
sim = nengo.Simulator(model)
sim.run(1.5)

fig = plt.figure()
plt.plot(sim.trange(), sim.data[input_probe], label="input")
plt.plot(sim.trange(), sim.data[ens_probe], label="ensemble")
plt.legend()
fig.savefig("3_e")