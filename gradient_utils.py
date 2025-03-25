import numpy as np
import scipy.optimize as so

# derivative precision
DERIVATIVE_DELTA = 10e-5

def __approx_derivative(func, x, dx):
    return (func(x + DERIVATIVE_DELTA * dx) - func(x)) / DERIVATIVE_DELTA


# flatten coefficients for tangents
TANGENT_FLATTEN_1 = 0.5
TANGENT_FLATTEN_2 = 0.8

# reducing step for alpha argument
Q_STEP = 0.9

# count of iterations for 1-dimensional methods
ITERATIONS = 50


def __one_dimensional_method(func_1, max_alpha, acceptable):
    f0 = func_1(0)
    df0 = __approx_derivative(func_1, 0, 1)
    soft_tangent_1 = lambda a: f0 + df0 * a * TANGENT_FLATTEN_1
    soft_tangent_2 = lambda a: f0 + df0 * a * TANGENT_FLATTEN_2
    alpha = max_alpha
    for _ in range(ITERATIONS):
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
    for _ in range(ITERATIONS):
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


# =====================


def __gradient(func_n, x):
    result = []
    size = len(x)
    for i in range(size):
        dxi = np.zeros(size)
        dxi[i] = 1
        result.append(__approx_derivative(func_n, x, dxi))
    return np.array(result)


def __draw_direction_3d(position, direction, value, plot):
    if np.linalg.norm(direction) < 1e-10:
        return
    plot.quiver(
        position[0], position[1], value,
        direction[0], direction[1], 0,
        linewidth=2, color='black', arrow_length_ratio=1, zorder=2
    )
    t = np.linspace(0, 1, 100)
    cx = position[0]
    cy = position[1]
    cz = t * value
    plot.plot(cx, cy, cz, linestyle='--', linewidth=1, color='black', zorder=0)


# count of gradient descent iterations
DESCENT_ITERATIONS = 100

# epsilon for stop condition
EPSILON = 1e-9

# epsilon for drawing gradients
DRAW_EPS = 0.5

def gradient_descent(
        func_n,
        min_values,
        max_values,
        start_position,
        next_position_strategy,
        plot
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

    :param plot: plot to draw anti-gradient directions

    :return: approximated minimum of func_n
    """

    current_x = start_position
    grad0_norm = np.linalg.norm(__gradient(func_n, current_x))**2

    for k in range(1, DESCENT_ITERATIONS + 1):
        grad = __gradient(func_n, current_x)
        if np.linalg.norm(grad) < EPSILON * grad0_norm:
            break
        __draw_direction_3d(current_x, -grad * DRAW_EPS, func_n(current_x), plot)
        func_1 = lambda a: func_n(current_x - a * grad) # restriction to the gradient line
        max_alpha = float('inf') # calculating how far we can move
        for i in range(len(grad)):
            if grad[i] < 0:
                max_alpha = min(max_alpha, (current_x[i] - max_values[i]) / grad[i])
            if grad[i] > 0:
                max_alpha = min(max_alpha, (current_x[i] - min_values[i]) / grad[i])
        step_c = next_position_strategy(k, func_1, max_alpha)
        current_x = current_x - step_c * grad

    return func_n(current_x)


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
        maxiter=ITERATIONS
    )
    alpha = result[0] if result is not None else None
    return alpha if alpha is not None else max_alpha * 0.5

goldstein_step_strategy = lambda k, fu, ma: __goldstein(fu, ma)
goldstein_advanced_step_strategy = lambda k, fu, ma: __goldstein_advanced(fu, ma)
