import matplotlib.pyplot as plt


def plot_points(ax, points, title):
    x = [i.x for i in points]
    y = [i.y for i in points]

    ax.scatter(x, y, s=10)
    ax.set_title(title)


def set_scale(resol, ax, pad=40):
    w = resol['width']
    h = resol['height']
    for ax_i in ax:
        ax_i.set_xlim(0-pad, w+pad)
        ax_i.set_ylim(0-pad, h+pad)


def plot_text(ax, word_aoi):
    for word_aoi_i in word_aoi:
        word_box = word_aoi_i.wordBox
        ax.text(word_box.x, word_box.y, word_aoi_i.word)
