import numpy as np
import random as rd

# plane
coord_sum = lambda _x: _x[0] + _x[1]

# qf strong conditioned
quadratic_strong_conditioned = lambda _x: _x[0] ** 2 + _x[1] ** 2

# qf ill conditioned
quadratic_ill_conditioned = lambda _x: 0.1 * _x[0] ** 2 + 10 * _x[1] ** 2

# like a cone function
norm_func = lambda _x: np.sqrt(_x[0] ** 2 + _x[1] ** 2)

# multimodal function, modal number can be increased by expanding domain
sinuses_sum = lambda _x: np.sin(_x[0]) + np.sin(_x[1])

# multimodal function
some_function = lambda _x: np.cos(np.sin(_x[0] ** 2 + _x[1]) + 1) * _x[0] + np.exp(np.cos(_x[1] ** 2))

# returns noisy function with parametrized noise
def noisy_func(noise_parameter):
    return lambda _x: _x[0] ** 2 + _x[1] ** 2 + rd.random() * noise_parameter

