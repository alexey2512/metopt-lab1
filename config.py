import numpy as np
import function_examples as fe
import gradient_utils as gu

# target function
function = fe.quadratic_3

# bounds
min_values = np.array([-5, -5])
max_values = np.array([5, 5])

#start position
start = np.array([1, 1])

# one dimensional strategy
strategy = gu.fixed_step_strategy(0.1)


__count = 400
__x = np.linspace(min_values[0], max_values[0], __count)
__y = np.linspace(min_values[1], max_values[1], __count)
X, Y = np.meshgrid(__x, __y)
Z = function(np.array([X, Y]))

points = None

def calculate():
    global points
    calc_count = 0
    def wrap(f):
        def inner(*args, **kwargs):
            nonlocal calc_count
            calc_count += 1
            return f(*args, **kwargs)
        return inner
    current_derivative_counter = gu.derivative_counter
    current_gradient_counter = gu.gradient_counter
    points = gu.gradient_descent(wrap(function), min_values, max_values, start, strategy)
    print('endpoint: ', points[-1])
    print('found value: ', function(np.array(points[-1])))
    print('descent iterations: ', len(points) - 1)
    print('function invocations: ', calc_count)
    print('derivative calculations: ', gu.derivative_counter - current_derivative_counter)
    print('gradient calculations: ', gu.gradient_counter - current_gradient_counter)
