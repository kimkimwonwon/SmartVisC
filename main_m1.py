import os
from utils.data_handler import DataHandler
import argparse
import matplotlib.pyplot as plt
from utils.visual import set_scale, plot_points


parser = argparse.ArgumentParser(
                    prog='SmartVC',
                    description='What the program does')

parser.add_argument("--is_sample",
                    type=str,
                    default="0",
                    )

args = parser.parse_args()


def compare_points(rp, point_cur, point_bm, resol):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    set_scale(resol, ax)
    plot_points(ax[0], rp)
    plot_points(ax[1], rp)
    plot_points(ax[0], point_cur, "current", c='blue', s=20, alpha=1)
    plot_points(ax[1], point_bm, "Benchmark", c='red', s=20, alpha=1)
    plt.show()


def main():
    # Phase-1 : iVT Filter & Line Allocation

    # Mission 1: iVT Filter
    # Raw gaze Point
    rp = handler.get_sample_rp()

    # Raw Fixation: sample
    bm_rf = handler.get_sample_rf()

    # Raw Fixation: 우리 알고리즘
    handler.run_ivt()
    current_rf = handler.get_sample_rf()

    # 그림으로 확인하기
    compare_points(rp, current_rf, bm_rf, handler.get_resolution())
    print()


if __name__ == '__main__':
    path_root = os.getcwd()
    handler = DataHandler(path_root, is_sample=True)
    main()
