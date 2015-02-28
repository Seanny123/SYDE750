import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import calc_rmse, rec_lin_neuron, lif_neuron, get_activities, get_decoders

def gen_rand_uc_vecs(dims, number):
	vecs = np.random.normal(size=(number,dims))
	print(vecs.shape)
	mags = np.linalg.norm(vecs, axis=-1)
	return vecs / mags[..., np.newaxis]

def gen_circle_points(points_num):
	points_list = []
	for i in range(points_num):
		t = 2*np.pi*np.random.uniform()
		r = np.sqrt(np.random.uniform())
		points_list.append([r*np.cos(t), r*np.sin(t)])
	return np.array(points_list)

def get_2d_activities(neuron_type, x_vals, n_neurons, x_cepts, max_firing_rates, gain_signs):
	neuron_list = []
	A = np.zeros( (n_neurons, x_vals[0].size) )
	for i in range(n_neurons):
		neuron_list.append(neuron_type(x_cepts[i], max_firing_rates[i]))
		A[i,:] = neuron_list[i](x_vals[0]*gain_signs[i][0] + x_vals[1]*gain_signs[i][1])
	return A, neuron_list

def get_2d_decoders(A, S, x_vals):
	gamma = np.dot(A, A.T) / S
	# so if I have a two dimensional upsilon....
	upsilon_0 = np.dot(A, x_vals[0]) / S
	upsilon_1 = np.dot(A, x_vals[1]) / S
	# I'll get two dimensional decoders....
	decoders_0 = np.dot(np.linalg.pinv(gamma), upsilon_0)
	decoders_1 = np.dot(np.linalg.pinv(gamma), upsilon_1)
	# Which I can just add up again to get the approximate value
	x_hat = np.dot(A.T, decoders_0) + np.dot(A.T, decoders_1)
	decoders = np.array([decoders_0, decoders_1])
	return decoders, x_hat

# generate random unit vectors and plot them
rand_vecs = gen_rand_uc_vecs(2, 100)
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

a = np.linspace(-1.0,1.0,100)
b = np.linspace(-1.0,1.0,100)

X,Y = np.meshgrid(a, b)

fig = plt.figure()
plt.plot(circle[0], circle[1])
for e in rand_vecs:
	plt.plot([0,e[0]], [0,e[1]], 'r')
fig.savefig("2_2a")

# generate some neurons
n_neurons = 100

max_firing_rates = np.random.uniform(100, 200, n_neurons)
x_cepts = np.random.uniform(-0.95, 0.95, n_neurons)
gain_signs = np.random.choice([-1, 1], n_neurons)
x_vals = [X.reshape((1,-1))[0], Y.reshape((1,-1))[0]]
A, neurons = get_2d_activities(
	lif_neuron,
	x_vals,
	n_neurons, x_cepts, max_firing_rates, rand_vecs)


# find some decoders
A_noisy = A.T + np.random.normal(scale=0.2*np.max(A), size=A.T.shape)
S = x_vals[0].size
decoders_noisy, x_hat_noisy = get_2d_decoders(A_noisy.T, S, x_vals)

test_vals = gen_circle_points(20)
new_A, neurons = get_2d_activities(
	lif_neuron,
	[ test_vals[:,0], test_vals[:,1] ],
	n_neurons, x_cepts, max_firing_rates, rand_vecs)
test_res = np.array([
	np.dot(new_A.T, decoders_noisy[0]), 
	np.dot(new_A.T, decoders_noisy[1])
])
print("rmse %s" %calc_rmse(test_res, test_vals.T))

# plot the decoders
fig = plt.figure()
for d_n in decoders_noisy.T:
	plt.plot([0,d_n[0]], [0,d_n[1]], 'r')
fig.savefig("2_2b_1")


# plot the decoded values
fig = plt.figure()
plt.plot(circle[0], circle[1])
plt.plot(test_vals[:,0], test_vals[:,1], 'ro', label="real values")
plt.plot(test_res[0], test_res[1], 'go', label="approximate values")
plt.legend(loc=4)
plt.title("Decoded values comparison")
fig.savefig("2_2b_2")


# decode with the encoders
enc_res = np.array([
	np.dot(new_A.T, rand_vecs[:,0]),
	np.dot(new_A.T, rand_vecs[:,1])
])
# just comparing angles
print("rmse with encoders %s" 
	%calc_rmse(
		np.arctan2(enc_res[0], enc_res[1]), 
		np.arctan2(test_vals[:,0], test_vals[:,1])
	)
)
print("rmse with normal decoders %s" 
	%calc_rmse(
		np.arctan2(test_res[0], test_res[1]),
		np.arctan2(test_vals[:,0], test_vals[:,1])
	)
)

ipdb.set_trace()
fig = plt.figure()
#plt.plot(test_vals[:,0], test_vals[:,1], 'ro', label="real values")
plt.plot(enc_res[0], enc_res[1], 'go', label="approximate values")
plt.legend(loc=4)
plt.title("Decoding with encoders")
fig.savefig("2_2b_3")
print("rmse with encoders %s" %calc_rmse(enc_res, test_vals.T))