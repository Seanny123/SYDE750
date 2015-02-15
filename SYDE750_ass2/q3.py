import numpy as np
import ipdb
import matplotlib.pyplot as plt
from utils import two_neurons, whitenoise

time_steps = np.arange(0, 1, 0.001)
neurons = two_neurons()
res_0 = []
for t in time_steps:
	res_0.append(neurons(0))

fig = plt.figure()
plt.title("Two neurons response to 0 input")
plt.plot(res_0)
plt.plot(np.zeros(len(res_0)))
plt.savefig("3_a")

neurons = two_neurons()
res_1 = []
for t in time_steps:
	res_1.append(neurons(1))

fig = plt.figure()
plt.title("Two neurons response to 1 input")
plt.plot(res_1)
plt.plot(np.ones(len(res_1)))
plt.savefig("3_b")

input_sig = lambda t: 0.5*np.sin(10*np.pi*t)
neurons = two_neurons()
res_sin = []
for t in time_steps:
	res_sin.append(neurons(input_sig(t)))

fig = plt.figure()
plt.title("Two neurons response to sin input")
plt.plot(res_sin)
plt.plot(input_sig(time_steps))
plt.savefig("3_c")

noise, _ = whitenoise(2, 0.001, 0.5, 5, 0)
neurons = two_neurons()
res_noise = []
for val in noise:
	res_noise.append(neurons(val))

fig = plt.figure()
plt.title("Two neurons response to noise input")
plt.plot(res_noise)
plt.plot(noise)
plt.savefig("3_d")