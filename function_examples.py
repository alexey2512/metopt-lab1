import numpy as np
import random as rd

# plane
coord_sum = lambda _x: _x[0] + _x[1]

# qf strong conditioned
quadratic_1 = lambda _x: _x[0] ** 2 + _x[1] ** 2

# qf strong conditioned stretched
quadratic_2 = lambda _x: 10 * _x[0] ** 2 + 10 * _x[1] ** 2

# qf ill conditioned
quadratic_3 = lambda _x: 2 * _x[0] ** 2 + 0.5 * _x[1] ** 2

# sin x cos y
sin_x_cos_y = lambda _x: np.sin(_x[0]) * np.cos(_x[1])

# non-continuous two paraboloids
min_of = lambda _x: np.minimum((_x[0] - 2) ** 2 + (_x[1] - 2) ** 2 + 2, _x[0] ** 2 + _x[1] ** 2)

# like a cone function
norm_func = lambda _x: np.sqrt(_x[0] ** 2 + _x[1] ** 2)

# multimodal function, modal number can be increased by expanding domain
sinuses_sum = lambda _x: np.sin(_x[0]) + np.sin(_x[1])

# multimodal function
some_function = lambda _x: np.cos(np.sin(_x[0] ** 2 + _x[1]) + 1) * _x[0] + np.exp(np.cos(_x[1] ** 2))

# happy cat
happy_cat = lambda _x: ((np.linalg.norm(_x) ** 2 - 2) ** 2) ** (1 / 8) + 0.5 * (0.5 * np.linalg.norm(_x) ** 2 + _x[0] + _x[1]) + 0.5

# returns noisy function with parametrized noise
def noisy_func(noise_parameter):
    return lambda _x: _x[0] ** 2 + _x[1] ** 2 + rd.random() * noise_parameter

