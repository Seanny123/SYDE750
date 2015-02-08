import numpy as np
import ipdb
import matplotlib.pyplot as plt

from utils import whitenoise

whitenoise(1.0, 0.001, rms=0.5, limit=5, seed=3)
whitenoise(4.0, 0.001, rms=0.5, limit=5, seed=3)