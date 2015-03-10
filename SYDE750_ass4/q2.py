import nengo
from nengo.dists import Uniform
from nengo.utils.ensemble import tuning_curves
import ipdb

model = nengo.Network()

with model:
	my_input = nengo.Node()
	ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002))
	out_ens = nengo.Ensemble(50, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002))
	nengo.Connection(ens, out_ens, synapse=0.01)


model = nengo.Network()

with model:
	my_input = nengo.Node()
	ens = nengo.Ensemble(100, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002))
	out_ens = nengo.Ensemble(50, dimensions=1, intercepts=Uniform(-1,1), max_rates=Uniform(100,200), neuron_type=nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002))
	nengo.Connection(ens, out_ens, function=lambda x: 2*x+1, synapse=0.01)