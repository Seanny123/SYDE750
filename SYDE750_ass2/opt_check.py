import numpy
import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise, two_neurons

T = 2.0         # length of signal in seconds
dt = 0.001      # time step size

def ideal_filter(T, dt, limit=5):
	# Generate bandlimited white noise (use your own function from part 1.1)
	x_vals = np.arange(0, T, dt)
	x = np.sin(4*T*x_vals)
	X = np.fft.fftshift(np.fft.fft(x))

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

pylab.figure()
pylab.plot(t, r, color='k', label='response', alpha=0.2)  # plot the spikes from the two neurons
pylab.plot(t, x, linewidth=2, label='signal')           # plot the input signal
pylab.plot(t, xhat, label='approximation')                     # plot the approximated signal
pylab.title('Comparison of Filter Result and Real Signal')
pylab.legend(loc='best')
pylab.xlabel('Time (s)')
pylab.savefig("opt_test")