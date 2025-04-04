import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import config as cfg

cfg.calculate()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(cfg.X, cfg.Y, cfg.Z, cmap=cm.coolwarm,
                     linewidth=0, antialiased=True,
                     alpha=0.7, rstride=10, cstride=10)

points_arr = np.array(cfg.points)

z_values = [cfg.function(p) for p in cfg.points]

ax.plot(points_arr[:,0], points_arr[:,1], z_values,
       'o-', color='black', markersize=5, linewidth=2,
       markerfacecolor='red', markeredgecolor='black')

ax.plot([cfg.points[0][0]], [cfg.points[0][1]], [z_values[0]],
       'o', color='green', markersize=10, label='Start')
ax.plot([cfg.points[-1][0]], [cfg.points[-1][1]], [z_values[-1]],
       'o', color='blue', markersize=10, label='End')

for i in range(0, len(cfg.points)-1, 3):
    ax.quiver(points_arr[i,0], points_arr[i,1], z_values[i],
             points_arr[i+1,0]-points_arr[i,0],
             points_arr[i+1,1]-points_arr[i,1],
             z_values[i+1]-z_values[i],
             color='blue', arrow_length_ratio=0.1, linewidth=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Gradient Descent Visualization')
ax.legend()

plt.tight_layout()
plt.show()
