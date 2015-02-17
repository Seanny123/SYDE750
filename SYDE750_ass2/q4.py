import numpy
import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, two_neurons

T = 2.0         # length of signal in seconds
dt = 0.001      # time step size

def ideal_filter(T, dt, limit=5):
	# Generate bandlimited white noise (use your own function from part 1.1)
	x, X = whitenoise(T, dt, rms=0.5, limit=limit, seed=3)
	X = np.fft.fftshift(X)

	Nt = len(x)                # number of time steps equal to the length of input
	t = numpy.arange(Nt) * dt  # generate the timestep by scaling a range by dt

	# Neuron parameters
	tau_ref = 0.002          # the refactory period of the neuron
	tau_rc = 0.02            # RC constant from circuit approximation of LIF neuron
	x0 = 0.0                 # firing rate at x=x0 is a0
	a0 = 40.0
	x1 = 1.0                 # firing rate at x=x1 is a1
	a1 = 150.0

	# calculate the parameters for the neurons
	eps = tau_rc/tau_ref
	r1 = 1.0 / (tau_ref * a0)
	r2 = 1.0 / (tau_ref * a1)
	f1 = (r1 - 1) / eps
	f2 = (r2 - 1) / eps
	alpha = (1.0/(numpy.exp(f2)-1) - 1.0/(numpy.exp(f1)-1))/(x1-x0)
	x_threshold = x0-1/(alpha*(numpy.exp(f1)-1))
	Jbias = 1-alpha*x_threshold

	# Simulate the two neurons (use your own function from part 3)
	neurons = two_neurons()
	spikes = []
	for val in x:
		spikes.append(neurons(val))
	spikes = np.array(spikes)


	freq = numpy.arange(Nt)/T - Nt/(2.0*T)   # create symmetrical frequency range
	omega = freq*2*numpy.pi                  # frequency to radians

	r = spikes[:, 0] - spikes[:, 1]                # response of the two neurons combined together
	R = numpy.fft.fftshift(numpy.fft.fft(r.T)) # transform response to frequency domain


	sigma_t = 0.025                          # the window size
	W2 = numpy.exp(-omega**2*sigma_t**2)     # create the gassian window
	W2 = W2 / sum(W2)                        # normalize the window size

	CP = X*R.conjugate()                  # ideal filter nominator before smoothing
	WCP = numpy.convolve(CP, W2, 'same')  # smooth the ideal filter with the gaussian window
	RP = R*R.conjugate()                  # the magnitude of the response spectrum
	WRP = numpy.convolve(RP, W2, 'same')  # smooth the magnitude of the response spectrum
	XP = X*X.conjugate()                  # the magnitude of the signal spectrum
	WXP = numpy.convolve(XP, W2, 'same')  # the magnitude of the signal spectrum smoothed with the gaussian window (not used)

	H = WCP / WRP                         # the optimal filter

	h = numpy.fft.fftshift(numpy.fft.ifft(numpy.fft.ifftshift(H))).real  # convert the filter to the time domain and only use the real parts

	XHAT = H*R                            # approximate the signal by convolving the response with the optimal filter

	xhat = numpy.fft.ifft(numpy.fft.ifftshift(XHAT)).real  # bring the approximate signal into the time domain
	return freq, XP, RP, H, h, r, x, xhat, XHAT, t

freq, XP, RP, H, h, r, x, xhat, XHAT, t = ideal_filter(T, dt)

import pylab

pylab.figure(1)
pylab.subplot(1,2,1)
pylab.plot(freq, numpy.sqrt(XP), label='Signal')  # plot the frequency spectrum of the input signal
pylab.plot(freq, XHAT, label='Approximation')
pylab.legend()
pylab.xlabel('$\omega$')
pylab.ylabel('$|X(\omega)|$')

pylab.subplot(1,2,2)
pylab.plot(freq, numpy.sqrt(RP), label='Response')  # plot the frequency spectrum of the response
pylab.legend()
pylab.xlabel('$\omega$')
pylab.ylabel('$|R(\omega)|$')
pylab.savefig("4_d")


pylab.figure(2)
pylab.subplot(1,2,1)
pylab.plot(freq, H.real)   # plot the optimal filter in the frequency domain
pylab.xlabel('$\omega$')
pylab.title('Optimal Filter in Frequency Domain')
pylab.xlim(-50, 50)

pylab.subplot(1,2,2)
pylab.plot(t-T/2, h)       # plot the optimal filter in the time domain
pylab.title('Optimal Filter in Time Domain')
pylab.xlabel('Time (s)')
pylab.xlim(-0.5, 0.5)
pylab.savefig("4_b")


pylab.figure(3)
pylab.plot(t, r, color='k', label='response', alpha=0.2)  # plot the spikes from the two neurons
pylab.plot(t, x, linewidth=2, label='signal')           # plot the input signal
pylab.plot(t, xhat, label='approximation')                     # plot the approximated signal
pylab.title('Comparison of Filter Result and Real Signal')
pylab.legend(loc='best')
pylab.xlabel('Time (s)')
pylab.savefig("4_c")

#pylab.show()

limit_list = [2, 10, 30]
h_list = []
for limit in limit_list:
	freq, XP, RP, H, h, r, x, xhat, XHAT, t = ideal_filter(T, dt, limit)
	h_list.append(h)

fig = plt.figure()
for i, h_val in enumerate(h_list):
	plt.plot(np.linspace(-h_val.size/2, h_val.size/2, h_val.size), h_val, label="limit=%s" %limit_list[i])
plt.legend()
plt.title("Effect of Changing Limit on Filter")
plt.xlabel("Time (s)")
plt.savefig("4_e")

# So what's the difference between how I'm filtering frequencies and how they're filtering frequencies?
T_list = [1.0, 4.0, 10.0]
H_list = []
h_list = []

for T_val in T_list:
	freq, XP, RP, H, h, r, x, xhat, XHAT, t = ideal_filter(T_val, dt)
	H_list.append(
		H[(H.size/2-400):(H.size/2+400)]
	)
	h_list.append(
		h[(h.size/2-200):(h.size/2+200)]
	)

omega = np.linspace(-400, 400, H_list[0].size)
time_vals = np.linspace(-200, 200, h_list[0].size)

fig = plt.figure()
for i, H_val in enumerate(H_list):
	plt.plot(omega/T_list[i], H_val, label="period=%s" %T_list[i])
plt.legend()
plt.title("Effect of Period Change on Filter in Frequency Domain")
plt.xlabel('$Frequency (Hz)$')
plt.savefig("4_f_1")

fig = plt.figure()
for i, h_val in enumerate(h_list):
	plt.plot(time_vals, h_val, label="period=%s" %T_list[i])
plt.legend()
plt.title("Effect of Period Change on Filter in Time Domain")
plt.xlabel("Time (s)")
plt.savefig("4_f_2")