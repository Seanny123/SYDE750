import numpy as np
import ipdb
import matplotlib.pyplot as plt
from utils import SpikingLif, modified_lif, whitenoise, spiking

dt = 0.001
lif_neuron = modified_lif(40, 150)
# try to make the spiking version work
spike = SpikingLif(dt=dt)
time_steps = np.arange(0, 1, dt)
res_0 = []
for t in time_steps:
	res_0.append(
		spiking(
			spike.spike(
				lif_neuron(
					0
				)
			)
		)
	)
print(spike.spike_count)
spike.spike_count = 0
res_1 = []
for t in time_steps:
	res_1.append(
		spiking(
			spike.spike(
				lif_neuron(
					1
				)
			)
		)
	)
print(spike.spike_count)
fig = plt.figure()
plt.plot(time_steps, res_0, label="x=0")
plt.plot(time_steps, res_1, label="x=1")
plt.legend()
plt.title("Spiking for one neuron, two inputs")
plt.savefig("2_a")

# what number of spikes are we expecting here anyways?
noise, _ = whitenoise(1, 0.001, 0.5, 30, 0)
noise_res = []
for val in noise:
	noise_res.append(
		spiking(
			spike.spike(
				lif_neuron(
					val
				)
			)
		)
	)

# this plot really looks terrible, like it's not responding to the crests and valleys at all. How do I make it look better?
fig = plt.figure()
plt.plot(noise)
plt.plot(noise_res)
plt.title("Spiking response to whitenoise")
plt.savefig("2_c")

spike.potential = 0
spike_res = []
for i in range(200):
	spike_res.append(
		spike.spike(
			lif_neuron(
				noise[i]
			)
		)
	)

fig = plt.figure()
plt.plot(noise[:200])
plt.plot(spike_res)
plt.title("Neuron potential")
plt.savefig("2_d")