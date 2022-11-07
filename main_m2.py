import os
import argparse
import matplotlib.pyplot as plt

from copy import deepcopy

import env
# from utils.metric import get_lineAllo_dashboard
from utils.data_handler import DataHandler
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
    word_aoi = handler.get_word_aoi()
    # Mission 1: iVT Filter
    # 이건 이미 성공했다고 가정
    handler.run_ivt()

    # Mission 2: Line Allocation
    bm_cf = deepcopy(handler.get_sample_cf())

    handler.run_alloc()
    current_cf = handler.get_sample_cf()

    # 그림으로 확인
    compare_points(current_cf, bm_cf, handler.get_word_aoi(), handler.get_resolution())

    # Metric
    # db = get_lineAllo_dashboard(handler.get_sample_rp(), word_aoi, current_cf, bm_cf)


if __name__ == '__main__':
    # 테스트할 때 여러 조건, 상태를 관리하는 방법으로 중간중간 상태를 확인하기 위한 plot을 다 보여주도록 설정한 것
    setattr(env, "SHOW_ALL_PLOTS", False)

    path_root = os.getcwd()
    handler = DataHandler(path_root, is_sample=False)
    main()
