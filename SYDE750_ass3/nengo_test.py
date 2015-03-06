import nengo

model = nengo.Network()

with model:
	nengo.Node()
	nengo.Ensemble(10)