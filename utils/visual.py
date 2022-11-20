from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import utils.const as const


def plot_points(ax, points, title="", c="black", s=10, alpha=0.5, is_save=False, fig=None):
    """
    ver.1.1: 그림그릴 때 점들이 시간 순서대로 그려나갈 때 어떻게 변화하는지 snapshot을 그리기 위해 is_save를 flag로 넣고, 그런 경우 각 점들에 대해서 그리고 저장하도록!
    """
    xs = [i.x for i in points]
    ys = [i.y for i in points]

    assert len(xs) == len(ys), f"Some points are missing, Length of x_coordinates : {len(xs)} & Length of y_coordinates : {len(ys)}"
    if not is_save:
        ax.scatter(xs, ys, s=s, c=c, alpha=alpha)
        ax.set_title(title)
    else:
        assert fig is not None, f"To save snapshot, fig argument is needed!"
        os.makedirs(f"figure/snapshot", exist_ok=True)
        ax.set_title(title)
        if type(s) != list:
            s = [s]*len(xs)
        else:
            assert len(xs) == len(s), f"Lengths of size list and point list are different!"
        prog = tqdm(enumerate(zip(xs, ys, s)))
        for i, (x, y, size) in prog:
            ax.scatter(x=x, y=y, s=size, c=c, alpha=alpha)

            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f"figure/snapshot/{title}_{i}th.png", bbox_inches=extent)

            prog.set_description(f"{i}th point of {len(xs)}")


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
        ax_i.invert_yaxis()


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
        ax.text(word_box.x, word_box.y, word_aoi_i.word, fontdict={"fontsize": const.font_size})

