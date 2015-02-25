import numpy as np
import ipdb
import matplotlib.pyplot as plt
from utils import whitenoise, two_neurons, calc_rmse, z_center

# assuming we won't plug in a negative t
def ptsc(t, n, tau):
	return ((t**n)*np.exp(-t/tau)) * (t > 0)

n_list = [0, 1, 2]
dt = 0.001
t_h = np.arange(1000)*dt-0.5

figure = plt.figure()
for n in n_list:
	res = ptsc(t_h, n, 0.007)
	res = res/np.linalg.norm(res)
	plt.plot(z_center(res), res, label="n=%s" %n)
plt.legend()
plt.savefig("5_a")

tau_list = [0.002, 0.005, 0.01, 0.02]
figure = plt.figure()
for tau in tau_list:
	res = ptsc(t_h, 0, tau)
	res = res/np.linalg.norm(res)
	plt.plot(z_center(res), res, label="tau=%s" %tau)
plt.legend()
plt.savefig("5_b")

noise, _ = whitenoise(1, dt, 0.5, 5, 0)
t_h = np.arange(998)*dt-0.5
neurons = two_neurons()
h = ptsc(t_h, 0, tau=0.007)
res_noise = []
for val in noise:
	res_noise.append(neurons(val))
res_noise = np.array(res_noise)

fspikes1 = np.convolve(res_noise[:,0], h, mode='same')
fspikes2 = np.convolve(res_noise[:,1], h, mode='same')

fig = plt.figure()
plt.plot(noise)
plt.plot(fspikes1)
plt.plot(fspikes2)
plt.savefig("spike_test")

A = np.array([fspikes1, fspikes2]).T
S = fspikes1.shape[0]

gamma = np.dot(A.T, A) / S
upsilon = np.dot(A.T, noise) / S
decoders = np.dot(np.linalg.pinv(gamma), upsilon)
x_hat = np.dot(A, decoders)

# Plot h(t)
figure = plt.figure()
plt.plot(z_center(h), h)
plt.savefig("5_c1")

# Plot h(t)
f_h = np.abs(np.fft.fftshift(np.fft.fft(h)))
figure = plt.figure()
plt.plot(z_center(f_h), f_h)
plt.savefig("5_c2")

# Plot x(t), the spikes, and x_hat
figure = plt.figure()
plt.plot(res_noise[:,0], label="neuron 1")
plt.plot(res_noise[:,1], label="neuron 2")
plt.plot(noise, label="signal")
plt.plot(x_hat, label="approximation")
plt.xlabel("time (s)")
plt.legend()
plt.savefig("5_c3")

# Try out the new decoders
new_noise, _ = whitenoise(1, dt, 0.5, 5, 1)
neurons = two_neurons()
res_noise = []
for val in new_noise:
	res_noise.append(neurons(val))
res_noise = np.array(res_noise)

fspikes1 = np.convolve(res_noise[:,0], h, mode='same')
fspikes2 = np.convolve(res_noise[:,1], h, mode='same')

A = np.array([fspikes1, fspikes2]).T
new_x_hat = np.dot(A, decoders)

print("original: %s" %calc_rmse(x_hat, noise))
print("new_sig: %s" %calc_rmse(new_x_hat, new_noise))