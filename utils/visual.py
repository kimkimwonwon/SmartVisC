import matplotlib.pyplot as plt


def plot_points(ax, points, title="", c="black", s=10, alpha=0.5):
    x = [i.x for i in points]
    y = [i.y for i in points]

    ax.scatter(x, y, s=s, c=c, alpha=alpha)
    ax.set_title(title)


def set_scale(resol, ax, pad=100):
    w = resol['width']
    h = resol['height']
    for ax_i in ax:
        ax_i.set_xlim(0-pad, w+pad)
        ax_i.set_ylim(0-pad, h+pad)


def show_line_plot(vals, title=""):
    plt.figure()
    plt.plot(vals)
    plt.title(title)
    plt.show()
