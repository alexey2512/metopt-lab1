import matplotlib.pyplot as plt
import config as cfg
import matplotlib.animation as animation

cfg.calculate()

index_p = -1

def data_gen():
    for i in range(len(cfg.points)):
        yield cfg.points[i][0], cfg.points[i][1]

def init():
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
CS = ax.contour(cfg.X, cfg.Y, cfg.Z, levels=25)
ax.clabel(CS, fontsize=7)
ax.set_title('Gradient Descent')
ax.set_xlabel('X')
ax.set_ylabel('Y')
line, = ax.plot([], [], lw=2, color='r')
ax.grid()
xdata, ydata = [], []

def run(data):
    xl, yl = data
    xdata.append(xl)
    ydata.append(yl)
    line.set_data(xdata, ydata)
    return line,

ani = animation.FuncAnimation(fig, run, data_gen, interval=300, init_func=init, save_count=min(len(cfg.points), 10))
ani.save('animation.gif', writer='pillow')
