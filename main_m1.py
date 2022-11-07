import os
import argparse
import matplotlib.pyplot as plt

from copy import deepcopy

import env
from utils.data_handler import DataHandler
# from utils.metric import get_ivt_dashboard
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
    plot_points(ax[0], point_cur, "current", c='blue', s=50, alpha=0.5)
    # ver.0.1 : 현재 데이터에 사에서 처리한 raw fixatio이 존재하지 않아서 같이 볼 수 없습니다
    # plot_points(ax[1], point_bm, "Benchmark", c='red', s=50, alpha=0.5)
    plt.show()


def main():
    # Phase-1 : iVT Filter & Line Allocation

    # Mission 1: iVT Filter
    # Raw gaze Point : plot에 사용할 점들이 필요. 초기 상태가 필요하기에 deepcopy 진행
    rp = deepcopy(handler.get_sample_rp())

    # Raw Fixation: sample
    bm_rf = handler.get_sample_rf()

    # Raw Fixation: 우리 알고리즘
    handler.run_ivt()
    current_rf = handler.get_sample_rf()

    # 그림으로 확인하기
    compare_points(rp, current_rf, bm_rf, handler.get_resolution())

    # Metric
    # db = get_ivt_dashboard(handler.get_sample_rp(), current_rf, bm_rf)


if __name__ == '__main__':
    # 테스트할 때 여러 조건, 상태를 관리하는 방법으로 중간중간 상태를 확인하기 위한 plot을 다 보여주도록 설정한 것
    setattr(env, "SHOW_ALL_PLOTS", False)

    path_root = os.getcwd()
    handler = DataHandler(path_root, is_sample=False)
    main()
