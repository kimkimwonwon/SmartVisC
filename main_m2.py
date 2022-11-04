import os
from utils.data_handler import DataHandler
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from utils.visual import set_scale, plot_points, plot_text


parser = argparse.ArgumentParser(
                    prog='SmartVC',
                    description='What the program does')

parser.add_argument("--is_sample",
                    type=str,
                    default="0",
                    )

args = parser.parse_args()


def compare_points(point_cur, point_bm, word_aoi, resol):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    set_scale(resol, ax)
    plot_points(ax[0], point_cur, "current", c="blue", s=50, alpha=0.5)
    plot_points(ax[1], point_bm, "Benchmark", c="red", s=50, alpha=0.5)

    plot_text(ax[0], word_aoi)
    plot_text(ax[1], word_aoi)
    plt.show()


def main():
    # Mission 1: iVT Filter
    # 이건 이미 성공했다고 가정
    handler.run_ivt()

    # Mission 2: Line Allocation
    bm_cf = deepcopy(handler.get_sample_cf())
    handler.run_alloc()
    current_cf = handler.get_sample_cf()
    compare_points(current_cf, bm_cf, handler.get_word_aoi(), handler.get_resolution())
    print()


if __name__ == '__main__':
    path_root = os.getcwd()
    handler = DataHandler(path_root, is_sample=True)
    main()
