import matplotlib.pyplot as plt
from gradient_utils import *
from function_examples import *

# choose function from function_examples
func_2 = some_function

# change domain and start position
min_values = np.array([-5, -5])
max_values = np.array([5, 5])
start_pos = np.array([0, 0])

# choose strategy from gradient_utils, u can choose lib analogues
next_position_strategy = fixed_step_strategy(1)


ax = plt.figure().add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fx = np.linspace(min_values[0], max_values[0], 100)
fy = np.linspace(min_values[1], max_values[1], 100)
fx, fy = np.meshgrid(fx, fy)
fz = func_2(np.array([fx, fy]))
ax.plot_surface(fx, fy, fz, color="yellow", alpha=0.7, zorder=1)

plt.scatter(start_pos[0], start_pos[1], func_2(np.array(start_pos)), linewidths=3, color='red', zorder=0)

result = gradient_descent(
    func_2,
    min_values,
    max_values,
    start_pos,
    next_position_strategy,
    ax
)

plt.show()

print(result, '\n\n')

bounds = [
    (min_values[0], max_values[0]),
    (min_values[1], max_values[1]),
]

# like armijo_advanced
lib_result_1 = so.minimize(func_2, start_pos, method='BFGS')
print(lib_result_1, '\n\n')

# with bounds
lib_result_2 = so.minimize(func_2, start_pos, method='L-BFGS-B', bounds=bounds)
print(lib_result_2, '\n\n')

