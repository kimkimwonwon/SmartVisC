import os
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from tabulate import tabulate

from copy import deepcopy

from utils.data import CorrectedFixation
from utils.data_handler import DataHandler
from utils.visual import set_scale, plot_points, plot_text, plot_lines
from utils.allocation import get_seg_thr, letter_and_gap, get_idx, word_cnt_mean
from utils import const
from main_m2 import duration_scaling
from collections import defaultdict
import params
import env


def calc_pad_width(word_aois):
    pad_width = [i.wordBox.width for i in word_aois]
    gap = get_seg_thr(word_aois, use_gap=True)

    for i in range(len(word_aois)):
        term = word_aois[i].wordBox.x - (word_aois[i - 1].wordBox.x + word_aois[i - 1].wordBox.width)
        if abs(term) > 10:      # line change 포인트
            pad_width[i - 1] = pad_width[i - 1] + gap / 2
        elif i == (len(word_aois) - 1):      # 마지막 단어
            pad_width[i] = pad_width[i] + gap / 2

    # padded x, y 좌표 (영점 기준 단어 우상단)
    xs = [(i.wordBox.x - gap / 2) for i in word_aois]
    pad_xs1 = [sum(x) for x in zip(xs, pad_width)]
    pad_ys1 = [(i.wordBox.y + gap / 2) for i in word_aois]
    pad_height = [(i.wordBox.height + gap) for i in word_aois]

    pad_xs2 = [xi - wi for xi, wi in zip(pad_xs1, pad_width)]
    pad_ys2 = [yi - hi for yi, hi in zip(pad_ys1, pad_height)]

    return pad_xs1, pad_ys1, pad_xs2, pad_ys2


def plot_pad_point(ax, handler, word_aois):
    pad_xs1, pad_ys1, pad_xs2, pad_ys2 = calc_pad_width(word_aois)

    gap = get_seg_thr(word_aois, use_gap=True)
    set_scale(handler.get_resolution(), ax)

    plot_text(ax[0], word_aois)
    plot_text(ax[1], word_aois)

    # 기존 x, y 좌표 (영점 기준 단어 좌상단)
    xs = [i.wordBox.x for i in word_aois]
    ys = [i.wordBox.y for i in word_aois]
    ax[0].scatter(xs, ys, c='green')
    ax[0].set_title("Original X, Y")

    # padded x, y 좌표
    ax[1].scatter(pad_xs1, pad_ys1, c='purple')     # 영점 기준 단어 좌하단
    ax[1].scatter(pad_xs2, pad_ys2, c='red')     # 영점 기준 단어 우상단
    ax[1].set_title(f"Padded X, Y (GAP = {gap})")

    plt.show()


def check_cf_in_word(cf, word_aois):
    # Check "fixation 수 >= (단어 수 * 3)"
    if len(cf) >= len(word_aois) * 3:
        print("Corrected Fixation 수가 충분합니다.")
    else:
        print("Corrected Fixation 수가 충분하지 않습니다!")
        print("Corrected Fixation 수 : ", len(cf))
        print("단어 수*3 : ", len(word_aois) * 3)

    # 단어 범위 내 fixation 수
    cf_x = [i.x for i in cf]
    cf_y = [i.y for i in cf]

    pad_xs1, pad_ys1, pad_xs2, pad_ys2 = calc_pad_width(word_aois)
    xs_in_word = []
    ys_in_word = []

    for i in range(len(pad_xs1)):
        x_in_word = []
        y_in_word = []

        for j in range(len(cf_x)):
            if (pad_xs2[i] <= cf_x[j] <= pad_xs1[i]) and (pad_ys2[i] <= cf_y[j] <= pad_ys1[i]):
                x_in_word.append(cf_x[j])
                y_in_word.append(cf_y[j])

        xs_in_word.append(x_in_word)
        ys_in_word.append(y_in_word)

    cnt_in_word = [len(i) for i in xs_in_word]

    return xs_in_word, ys_in_word, cnt_in_word


# 확인용
# # X, Y vs. padded X, Y plot
# # 한글 깨짐 ('AppleGothic'은 자간이 넓어 사용 불가)
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
#
# os.chdir('..')
# path_root = os.getcwd()
# handler = DataHandler(path_root)
# word_aois = handler.get_word_aoi()
#
# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# plot_pad_point(ax, handler, word_aois)
#
#
# # X, Ys in padded word area
# cf = handler.get_sample_cf()
# xs_in_word, ys_in_word, cnt_in_word = check_cf_in_word(cf, word_aois)
#
# print(xs_in_word)
# print(ys_in_word)
# print(cnt_in_word)
# print("단어 범위 내 fixation 수 평균 :", np.mean(cnt_in_word))


# Use get_seg_thr instead!
# Estimating Word Gap
# # font_size = 42.75
# # estim_width = []
# pad_width = [i.wordBox.width for i in word_aoi]
# gap = 2.25
#
# print("The Number of Word : ", len(word_aoi))

# for i in range(len(word_aoi)):
#     print("Word : ", word_aoi[i].word)
#     print("WordBox : ", word_aoi[i].wordBox)
#
#     # estim_width.append(word_aoi[i].wordBox.width - (word_aoi[i].word_cnt-1)*font_size)
#
#     if i > 0:
#         """
#         i번째 단어 왼쪽 끝 x좌표 - i-1번째 단어 오른쪽 끝 x좌표
#         = i번째 단어 x좌표 - (i-1번째 단어 x좌표 + 해당 width)
#         이게 대체로 다음 단어 x좌표랑 동일... term은 word_cnt 이용하는 estim_width로 계산
#         """
#         #
#         # print("Calc. Term ((i-1)th x+w) : ", word_aoi[i-1].wordBox.x + word_aoi[i-1].wordBox.width)
#         print("---------------")
#
#         term = (word_aoi[i].wordBox.x) - (word_aoi[i-1].wordBox.x + word_aoi[i-1].wordBox.width)
#         if abs(term) > 10:
#             print("! 줄 바꿈 발생 포인트 !")
#             pad_width[i-1] = pad_width[i-1] + gap/2
#         elif i == (len(word_aoi)-1):
#             print("! 마지막 단어 !")
#             pad_width[i] = pad_width[i] + gap/2
#
#     print("---------------")

# print(np.median(estim_width))
# print(np.mean(estim_width))

