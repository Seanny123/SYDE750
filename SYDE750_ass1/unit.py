import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def gen_rand_vecs(dims, number):
	vecs = np.random.uniform(low=-1, size=(number,dims))

def gen_rand_uc_vecs(dims, number):
	vecs = np.random.normal(size=(number,dims))
	mags = np.linalg.norm(vecs, axis=-1)
	return vecs / mags[..., np.newaxis]

theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

fig = plt.figure()
plt.plot(circle[0], circle[1])
rand_vecs = gen_rand_vecs(2, 100)
for e in rand_vecs:
	plt.plot([0,e[0]], [0,e[1]], 'r')
plt.show()