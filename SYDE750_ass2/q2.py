import numpy as np
import ipdb
import matplotlib.pyplot as plt
from utils import SpikingLif, modified_lif, whitenoise

lif_neuron = modified_lif(40, 150)
# try to make the spiking version work
spike = SpikingLif()
time_steps = np.arange(0, 1, 0.001)
res_0 = []
for t in time_steps:
	res_0.append(
		(spike.spike(
			lif_neuron(
				np.array([0])
			)[0]
		) < 2)
	)
print(spike.spike_count)
spike.spike_count = 0
res_1 = []
for t in time_steps:
	res_1.append(
		(spike.spike(
			lif_neuron(
				np.array([1])
			)[0]
		) < 2)
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
		(spike.spike(
			lif_neuron(
				np.array([val])
			)[0]
		) < 2)
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
				np.array([noise[i]])
			)[0]
		)
	)

fig = plt.figure()
plt.plot(noise[:200])
plt.plot(spike_res)
plt.title("Neuron potential")
plt.savefig("2_d")