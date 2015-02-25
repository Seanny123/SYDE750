import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, calc_rmse, lif_ensemble, modified_lif, z_center, ptsc, get_decoders

dt = 0.001
t_range = np.arange(998)*dt-0.5
h = ptsc(t_range, 0.005)
fig = plt.figure()
plt.plot(z_center(h), h)
plt.title("post-synaptic current filter")
plt.savefig("2_a")

# create the noise signal
noise_sig, _ = whitenoise(1, dt, 1, 5, 0)

# create the lif ensemble
lifs = []
lifs.append(modified_lif(0.2, 150))
lifs.append(modified_lif(0.2, 150))
ensemble = lif_ensemble(lifs, [-1, 1])

# get the spikes
res_noise = []
for val in noise_sig:
	res_noise.append(ensemble(val))
res_noise = np.array(res_noise)

# first let's make sure the spikes actually makes sense
fig = plt.figure()
plt.plot(res_noise)
plt.plot(noise_sig)
plt.savefig("spike_test")

# filter the spikes
A = np.zeros((t_range.size, res_noise.shape[1]))
for i_n in range(len(lifs)):
	A[:,i_n] = np.convolve(res_noise[:,i_n], h, mode='same')

# after filtering
fig = plt.figure()
plt.plot(A)
plt.plot(noise_sig)
plt.savefig("blur_test")

# decode with the normal LIF encoders
_, x_hat = get_decoders(A.T, A.shape[0], noise_sig)

fig = plt.figure()
plt.plot(noise_sig)
plt.plot(x_hat)
plt.plot(res_noise)
plt.title("result of filtered spikes")
plt.savefig("2_b")

print("rmse:%s" %str(calc_rmse(x_hat, noise_sig)))