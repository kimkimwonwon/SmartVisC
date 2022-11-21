import os
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from copy import deepcopy

from utils.data import CorrectedFixation
from utils.data_handler import DataHandler
from utils.visual import set_scale, plot_points, plot_text, plot_lines
from utils import const
from main_m2 import duration_scaling
from collections import defaultdict
import params
import env


def calc_pad_width(word_aoi, gab=2.25):
    pad_width = [i.wordBox.width for i in word_aoi]

    for i in range(len(word_aoi)):
        term = (word_aoi[i].wordBox.x) - (word_aoi[i - 1].wordBox.x + word_aoi[i - 1].wordBox.width)
        if abs(term) > 10:      # 줄 바꿈 발생 포인트
            pad_width[i - 1] = pad_width[i - 1] + gab / 2
        elif i == (len(word_aoi) - 1):      # 마지막 단어
            pad_width[i] = pad_width[i] + gab / 2

    # padded x, y 좌표 (영점 기준 단어 우상단)
    xs = [(i.wordBox.x - gab / 2) for i in word_aoi]
    pad_xs = [sum(x) for x in zip(xs, pad_width)]

    return pad_width, pad_xs


def plot_pad_point(ax, word_aoi, pad_xs, gab=2.25):
    set_scale(handler.get_resolution(), ax)

    plot_text(ax[0], word_aoi)
    plot_text(ax[1], word_aoi)

    # 기존 x, y 좌표 (영점 기준 단어 좌상단)
    xs = [i.wordBox.x for i in word_aoi]
    ys = [i.wordBox.y for i in word_aoi]
    ax[0].scatter(xs, ys, c='green')
    ax[0].set_title("Original X, Y")

    # padded x, y 좌표 (영점 기준 단어 우상단)
    ax[1].scatter(pad_xs, ys, c='purple')
    ax[1].set_title(f"X-1/2GAB (GAB = {gab}), Y")

    plt.show()


# x, y좌표 plot
# 한글 깨짐 ('AppleGothic'은 자간이 넓어 사용 불가)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Apple SD Gothic Neo'

os.chdir('..')
path_root = os.getcwd()
handler = DataHandler(path_root)
word_aoi = handler.get_word_aoi()

pad_width, pad_xs = calc_pad_width(word_aoi)
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
plot_pad_point(ax, word_aoi, pad_xs)


# Estimating Word Gab
# # font_size = 42.75
# # estim_width = []
# pad_width = [i.wordBox.width for i in word_aoi]
# gab = 2.25
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
#             pad_width[i-1] = pad_width[i-1] + gab/2
#         elif i == (len(word_aoi)-1):
#             print("! 마지막 단어 !")
#             pad_width[i] = pad_width[i] + gab/2
#
#     print("---------------")

# print(np.median(estim_width))
# print(np.mean(estim_width))

