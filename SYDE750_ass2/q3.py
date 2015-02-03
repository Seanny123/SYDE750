# so figure out the voltage as a function of current
# the amount of steps should be the division of your x values, plugged into that current equation, until you reach the spiking level
# the step size should be
def euler_method(f, x0, y0, h, n):

# use the result from LIF spiking euler method to check if the threshold is reached
# once the threshold is reached, spike
def lif_spiking(t_rc, t_ref):
