import numpy as np
import ipdb
import matplotlib.pyplot as plt

np.random.seed(0)
# create 6 coefficients
coef = np.random.normal(0, 1, 4) + 1j * np.random.normal(0, 1, 4)
# set DC to zero
coef[0] = 0
# now try to mirror it using your method
if(coef.size % 2 == 1):
	print("odd")
	final_coef = np.zeros(coef[1:].size * 2 + 1, dtype=np.complex_)
	final_coef[coef.size] = coef
	final_coef[coef.size:] = coef[1:][::-1].conj()
else:
	print("even")
	final_coef = np.zeros(coef[1:-1].size * 2 + 2, dtype=np.complex_)
	# Don't touch the DC or the middle term!
	final_coef[:coef.size] = coef
	final_coef[coef.size:] = coef[1:-1][::-1].conj()

ipdb.set_trace()