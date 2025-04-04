import matplotlib.pyplot as plt
import config as cfg
from matplotlib.patches import FancyArrow
from matplotlib.colors import LinearSegmentedColormap

cfg.calculate()

fig, ax = plt.subplots()
CS = ax.contour(cfg.X, cfg.Y, cfg.Z, levels=25)
ax.clabel(CS, fontsize=7)
ax.grid()
ax.set_title('2D Gradient Descent Visualization')
ax.set_xlabel('X')
ax.set_ylabel('Y')

cmap = LinearSegmentedColormap.from_list('gradient', ['red', 'blue'])
n_points = len(cfg.points)
color_norm = plt.Normalize(0, n_points-2)

for i in range(n_points - 1):
    dx = cfg.points[i + 1][0] - cfg.points[i][0]
    dy = cfg.points[i + 1][1] - cfg.points[i][1]
    arrow = FancyArrow(cfg.points[i][0], cfg.points[i][1], dx, dy,
                       width=0.01, length_includes_head=True,
                       head_width=0.15, head_length=0.15,
                       color=cmap(color_norm(i)), alpha=1)
    ax.add_patch(arrow)

plt.show()
