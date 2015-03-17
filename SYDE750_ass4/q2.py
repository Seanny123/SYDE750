import nengo
from nengo.dists import Uniform
import matplotlib.pyplot as plt
from nengo.utils.functions import piecewise
import ipdb

model = nengo.Network()

with model:
	my_input = nengo.Node(piecewise({0:1, 0.1:0.0}))
	ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002))
	out_ens = nengo.Ensemble(50, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002))
	nengo.Connection(my_input, ens)
	nengo.Connection(ens, out_ens, synapse=0.01)

	input_probe = nengo.Probe(my_input)
	p1 = nengo.Probe(ens)
	p2 = nengo.Probe(out_ens)

sim = nengo.Simulator(model)
sim.run(0.5)

fig = plt.figure()
plt.title("input function")
plt.plot(sim.trange(), sim.data[input_probe])
fig.savefig("2_a_1")

fig = plt.figure()
plt.title("first ensemble")
plt.plot(sim.trange(), sim.data[p1])
fig.savefig("2_a_2")

fig = plt.figure()
plt.title("second ensemble")
plt.plot(sim.trange(), sim.data[p2])
fig.savefig("2_a_3")

model = nengo.Network()

def connection_function(x):
	return 1-2*x

with model:
	my_input = nengo.Node(1)
	ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002))
	out_ens = nengo.Ensemble(50, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002))
	nengo.Connection(my_input, ens)
	nengo.Connection(ens, out_ens, function=connection_function, synapse=0.01)

	input_probe = nengo.Probe(my_input)
	p1 = nengo.Probe(ens)
	p2 = nengo.Probe(out_ens)

sim = nengo.Simulator(model)
sim.run(0.5)

fig = plt.figure()
plt.title("input function")
plt.plot(sim.trange(), sim.data[input_probe])
fig.savefig("2_b_1")

fig = plt.figure()
plt.title("first ensemble")
plt.plot(sim.trange(), sim.data[p1])
fig.savefig("2_b_2")

fig = plt.figure()
plt.title("second ensemble")
plt.plot(sim.trange(), sim.data[p2])
fig.savefig("2_b_3")