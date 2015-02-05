import numpy as np
import ipdb
import matplotlib.pyplot as plt
from utils import two_neurons

time_steps = np.arange(0, 1, 0.001)
neurons = two_neurons()
res_0 = []
for t in time_steps:
	res_0.append(neurons(0))

fig = plt.figure()
plt.title("Two neurons response to 0 input")
plt.plot(res_0)
plt.plot(np.zeros(res_0.shape))
plt.savefig("3_a")

neurons = two_neurons()
res_1 = []
for t in time_steps:
	res_1.append(neurons(1))

fig = plt.figure()
plt.title("Two neurons response to 1 input")
plt.plot(res_1)
plt.plot(np.zeros(res_1.shape))
plt.savefig("3_b")

input_sig = lambda t: 0.5*np.sin(10*np.pi*t)