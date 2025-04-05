import numpy as np
import scipy.optimize as so

# derivative precision
DERIVATIVE_DELTA = 10e-5

# global counter of derivative invocations
derivative_counter = 0

def __approx_derivative(func, x, dx, consider=True):
    global derivative_counter
    if consider:
        derivative_counter += 1
    return (func(x + DERIVATIVE_DELTA * dx) - func(x - DERIVATIVE_DELTA * dx)) / DERIVATIVE_DELTA / 2.0

# flatten coefficients for tangents
TANGENT_FLATTEN_1 = 1e-3
TANGENT_FLATTEN_2 = 0.7

# reducing step for alpha argument
Q_STEP = 0.9

# count of iterations for 1-dimensional methods
ONE_DIM_ITERATIONS = 100

def __one_dimensional_method(func_1, max_alpha, acceptable):
    f0 = func_1(0)
    df0 = __approx_derivative(func_1, 0, 1)
    soft_tangent_1 = lambda a: f0 + df0 * a * TANGENT_FLATTEN_1
    soft_tangent_2 = lambda a: f0 + df0 * a * TANGENT_FLATTEN_2
    alpha = max_alpha
    for _ in range(ONE_DIM_ITERATIONS):
        if acceptable(func_1, alpha, soft_tangent_1, soft_tangent_2, df0):
            break
        alpha = alpha * Q_STEP
    return alpha


# selects argmin among acceptable points
def __one_dimensional_method_advanced(func_1, max_alpha, acceptable):
    f0 = func_1(0)
    df0 = __approx_derivative(func_1, 0, 1)
    soft_tangent_1 = lambda a: f0 + df0 * a * TANGENT_FLATTEN_1
    soft_tangent_2 = lambda a: f0 + df0 * a * TANGENT_FLATTEN_2
    alpha = max_alpha
    min_alpha = alpha
    f_min_alpha = func_1(min_alpha)
    for _ in range(ONE_DIM_ITERATIONS):
        if acceptable(func_1, alpha, soft_tangent_1, soft_tangent_2, df0):
            f_alpha = func_1(alpha)
            if f_alpha < f_min_alpha:
                min_alpha = alpha
                f_min_alpha = f_alpha
        alpha = alpha * Q_STEP
    return min_alpha if f_min_alpha < func_1(alpha) else alpha


def __armijo_acceptable(func_1, alpha, soft_tangent_1, soft_tangent_2, df0):
    return func_1(alpha) < soft_tangent_1(alpha)

def __armijo(func_1, max_alpha):
    return __one_dimensional_method(func_1, max_alpha, __armijo_acceptable)

def __armijo_advanced(func_1, max_alpha):
    return __one_dimensional_method_advanced(func_1, max_alpha, __armijo_acceptable)

def __wolfe_strategy(func_1, alpha, soft_tangent_1, soft_tangent_2, df0):
    dfa = __approx_derivative(func_1, alpha, 1)
    threshold_alpha = df0 * TANGENT_FLATTEN_2
    return dfa > threshold_alpha and func_1(alpha) < soft_tangent_1(alpha)

def __wolfe(func_1, max_alpha):
    return __one_dimensional_method(func_1, max_alpha, __wolfe_strategy)

def __wolfe_advanced(func_1, max_alpha):
    return __one_dimensional_method_advanced(func_1, max_alpha, __wolfe_strategy)

def __goldstein_strategy(func_1, alpha, soft_tangent_1, soft_tangent_2, df0):
    return soft_tangent_2(alpha) < func_1(alpha) < soft_tangent_1(alpha)

def __goldstein(func_1, max_alpha):
    return __one_dimensional_method(func_1, max_alpha, __goldstein_strategy)

def __goldstein_advanced(func_1, max_alpha):
    return __one_dimensional_method_advanced(func_1, max_alpha, __goldstein_strategy)

#global counter of gradient invocations
gradient_counter = 0

def __gradient(func_n, x, consider=True):
    global gradient_counter
    if consider:
        gradient_counter += 1
    result = []
    size = len(x)
    for i in range(size):
        dxi = np.zeros(size)
        dxi[i] = 1
        result.append(__approx_derivative(func_n, x, dxi, consider))
    return np.array(result)

# count of gradient descent iterations
DESCENT_ITERATIONS = 100

# epsilon for stop condition
EPSILON = 1e-5

def gradient_descent(
        func_n,
        min_values,
        max_values,
        start_position,
        next_position_strategy
):
    """
    Executes a gradient descent algorithm from start_position
    using next_position_strategy to get the next position.
    Supposed to be used for 2-dimensional functions.

    :param func_n: n-dimensional function to R

    :param min_values: lower bounds for arguments of func_n

    :param max_values: higher bounds for arguments of func_n

    :param start_position: start position of algorithm

    :param next_position_strategy: function from (k, func_1, max_alpha) where
        k - number of iteration,
        func_1(a) = func_n(x_k - a * grad),
        max_alpha = higher bound for argument of func_1,
    which returns alpha to get next position by formula: x_k - alpha * grad

    :return: approximated minimum of func_n
    """

    current_x = start_position
    grad0_norm = np.linalg.norm(__gradient(func_n, current_x, False))**2
    positions = [start_position]
    for k in range(1, DESCENT_ITERATIONS + 1):
        grad = __gradient(func_n, current_x)
        if np.linalg.norm(grad) < EPSILON * grad0_norm:
            break
        func_1 = lambda a: func_n(current_x - a * grad)
        max_alpha = float('inf')
        for i in range(len(grad)):
            if grad[i] < 0:
                max_alpha = min(max_alpha, (current_x[i] - max_values[i]) / grad[i])
            if grad[i] > 0:
                max_alpha = min(max_alpha, (current_x[i] - min_values[i]) / grad[i])
        step_c = next_position_strategy(k, func_1, max_alpha)
        current_x = current_x - step_c * grad
        positions.append(current_x)
    return positions


# ==================
# example strategies

def fixed_step_strategy(step):
    return lambda k, fu, ma: min(ma, step)

def dynamic_step_strategy(step_from_k):
    return lambda k, fu, ma: min(ma, step_from_k(k))

armijo_step_strategy = lambda k, fu, ma: __armijo(fu, ma)
armijo_advanced_step_strategy = lambda k, fu, ma: __armijo_advanced(fu, ma)

# lib analogue of armijo, use it like a strategy
def armijo_step_strategy_lib(_, func_1, max_alpha):
    phi0 = func_1(0)
    derphi0 = -np.linalg.norm(func_1(0) - func_1(1)) ** 2
    alpha, _ = so.linesearch.scalar_search_armijo(
        phi=lambda a: func_1(a),
        phi0=phi0,
        derphi0=derphi0,
        alpha0=min(1.0, max_alpha)
    )
    return min(alpha, max_alpha) if alpha is not None else max_alpha * 0.1

wolfe_step_strategy = lambda k, fu, ma: __wolfe(fu, ma)
wolfe_advanced_step_strategy = lambda k, fu, ma: __wolfe_advanced(fu, ma)

# lib analogue of wolfe, use like a strategy
def wolfe_step_strategy_lib(_, func_1, max_alpha):
    phi = func_1
    derphi = lambda a: __approx_derivative(phi, a, 1)
    phi0 = phi(0)
    derphi0 = derphi(0)
    result = so.linesearch.scalar_search_wolfe2(
        phi, derphi,
        phi0=phi0, derphi0=derphi0,
        c1=TANGENT_FLATTEN_1, c2=TANGENT_FLATTEN_2,
        maxiter=ONE_DIM_ITERATIONS
    )
    alpha = result[0] if result is not None else None
    return alpha if alpha is not None else max_alpha * 0.5

goldstein_step_strategy = lambda k, fu, ma: __goldstein(fu, ma)
goldstein_advanced_step_strategy = lambda k, fu, ma: __goldstein_advanced(fu, ma)
