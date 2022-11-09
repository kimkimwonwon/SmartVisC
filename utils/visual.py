import matplotlib.pyplot as plt


def plot_points(ax, points, title="", c="black", s=10, alpha=0.5):
    x = [i.x for i in points if i.x]
    y = [i.y for i in points]

    ax.scatter(x, y, s=s, c=c, alpha=alpha)
    ax.set_title(title)


def plot_lines(ax, points, c="black", lw=1):
    x = [i.x for i in points if i.x]
    y = [i.y for i in points]

    ax.plot(x, y, c=c, lw=lw)


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


def show_line_plot_compare(before, after, title):
    plt.figure(figsize=(14, 7))
    plt.plot(before, label="Before", lw=1, c="blue", alpha=0.5)
    plt.plot(after, label="After", lw=1, c="red", alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.show()


def plot_text(ax, word_aoi):
    for word_aoi_i in word_aoi:
        word_box = word_aoi_i.wordBox
        ax.text(word_box.x, word_box.y, word_aoi_i.word)

