"""
[Mission 1]
Based on the traditional(or conventional) iVT filter process, customize it!
    - Raw Gaze Point --> Raw Fixation


Notation:
    - RP: Raw gaze Point
    - RF : Raw Fixation
"""

from utils.data import RawFixation
from scipy import signal
import numpy as np
import env
import params
from collections import defaultdict
from utils.visual import show_line_plot
from sklearn.metrics import pairwise_distances


def get_xs(obj: list):
    return np.array([i.x for i in obj])


def get_ys(obj: list):
    return np.array([i.y for i in obj])


def gap_fill_in(rps):
    # TBD: Gap fill in 구현이 이미 되어 있는 데이터의 상태. 추후에 데이터를 받게 되면 구현해야할 필요 가능
    # Parameter: Max Gap Length as mgl
    """
    :param rps: List<RawGazePoint>
    :return: List<RawGazePoint>
    다만 여기서 rps의 내부에서 값들의 업데이트나 변경사항이 있다면 알려주기
    ex. null value를 어떻게 interpolation을 했다
    """
    mgl = params.max_gap_length
    return rps


def noise_reduction(rps):
    # Parameter: Window size for Moving average
    """
    :param rps: List<RawGazePoint>
    :return: List<RawGazePoint>
    Baseline으로 현재 Moving average를 한 값으로 기존의 값을 업데이트 진행
        - 사실 신호에 대해 convolution연산을 하는 것과 비슷한 것
    """
    xs = get_xs(rps)
    ys = get_ys(rps)

    # 방법 1: avg
    window_size = params.window_size
    window = np.ones(window_size)/window_size
    xs_new = signal.convolve(xs, window, mode="same")
    ys_new = signal.convolve(ys, window, mode="same")
    for rp, x, y in zip(rps, xs_new, ys_new):
        rp.x = x
        rp.y = y
    return rps


def calculate_velocity(rps):
    # Parameter : window length(baseline에는 아직 적용하지 않음)
    """
    :param rps: List<RawGazePoint>
    :return: List<RawGazePoint>
    Baseline으로 단순히 1step간의 차이를 속력으로 계산 및 업데이트
    다만 이 속력 계산시 구간에 대한 parameter를 어떻게 적용하는지에 따라서 달라질 것 같은데
    해당 내용은 완벽히 숙지한 것은 아니에요..! 그래서 구현하실 때 해당 파트 디테일하게 보고 적용해도 될 것 같습니다.
    """

    window_len = params.window_len

    xs = get_xs(rps)
    ys = get_ys(rps)

    delta_x = np.lib.stride_tricks.sliding_window_view(xs, 2, writeable=True)
    delta_x = delta_x[:, 1] - delta_x[:, 0]
    delta_y = np.lib.stride_tricks.sliding_window_view(ys, 2, writeable=True)
    delta_y = delta_y[:, 1] - delta_y[:, 0]
    # NOTE: 그런데, 여기서 속력이 아니라 속도를 구한다면..?!
    speeds = np.abs(delta_y / delta_x)
    # NA 값 처리
    # TODO: 어떻게 값을 처리할지 정해야 함. 시간 계산을 하게 되면 1step이 사라지게 되어서
    speeds = np.append(speeds, [0])

    # 각 point에 대해 속력값 업데이트
    for rp, speed in zip(rps, speeds):
        setattr(rp, "speed", speed)

    if env.SHOW_ALL_PLOTS:
        show_line_plot(speeds)
    return rps


def absolute_threshold(rps):
    # Parameter : Velocity Threshold
    """
    :param rps: List<RawGazePoint>
    :return: List<RawGBazePoint>
    Baseline으로 단순히 해당 threshold를 넘으면 saccade 아래면 fixation으로 설정
    """
    for rp in rps:
        if rp.speed > params.velocity_threshold:
            setattr(rp, "label", "saccade")
        else:
            setattr(rp, "label", "fixation")
    return rps


def merge_adj_fixation(rps):
    # Parameter: Max time between fixations
    # 추가로 cluster threshold, The Number of neighbors 를 사용
    """
    :param rps: List<RawGazePoint>
    :return: List<RawGazePoint>
    Fixation 사이 Max time: 지금은 죄다 33, 34라서 문제라서 Deprecated된 코드는 사용하지 못하고 있음
    새로 데이터를 받았을 때 확인해봐야 하는 부분.

    Baseline으로 인접하다고 묶인 경우에 대해서 fix_group_id라는 attr를 업데이트 함!
    즉 각 작은 점들마다 합쳐서 하나의 점들로 생각할 경우에는 동일한 fix_group_id를 가지도록

    cluste
    """

    # # Deprecated
    # times = np.array([int(i.timestamp) for i in rps])
    # delta_t = np.lib.stride_tricks.sliding_window_view(times, 2, writeable=True)
    # delta_t = delta_t[:, 1] - delta_t[:, 0]
    # fix_group_id = (delta_t > params.max_time_between_fixations).astype(int).cumsum()

    ###########################################################################
    # Euclidean distance를 기준으로 묶는 과정
    # 다만 어떤 기준으로 묶는지는 임시로 구현해놓은 것

    cluster_thr = params.cluster_thr
    num_neighbor = params.num_neighbor

    xs = get_xs(rps)[:, np.newaxis]
    ys = get_ys(rps)[:, np.newaxis]
    coors = np.concatenate((xs, ys), axis=1)
    distance_mat = pairwise_distances(coors)

    # 임시로 정한 metric
    nth_neighbor = np.sort(distance_mat, axis=1)[:, 1]/np.sort(distance_mat, axis=1)[:, 1+num_neighbor]
    fix_group_id = (nth_neighbor > cluster_thr).astype(int).cumsum()

    # NA 값 처리
    fix_group_id = np.append(fix_group_id, fix_group_id[-1])
    ###########################################################################

    # 값 업데이트
    for rp, group_id in zip(rps, fix_group_id):
        setattr(rp, "fix_group_id", group_id)
    return rps


def discard_short_fixation(rps):
    # Parameter : Minimum fixation parameter as min_fix_duration
    """
    :param rps: List<RawGazePoint>
    :return: List<RawFixation>

    해당 과정에서 Raw Fixation을 반환하게 됨.
    Baseline에서는 그냥 merge해놓은 동일한 group만 반환하고 있고,
    실제로 거리에 따라서 버리는 것은 구현되어 있지 않음.

    RawFixation을 만들기 위해선
    (x, y, timestamp, duration)이 필수요소로, 이를 만들어 주는 코드가 같이 구현되어야 함.
    """
    min_fix_duration = params.min_fix_duration

    # NOTE: 전처리된 rps에서 이제 골라서 duration까지 구하는 것 포함
    fix_groups = defaultdict(lambda: defaultdict(list))
    for rp in rps:
        fix_groups[rp.fix_group_id]["x"].append(rp.x)
        fix_groups[rp.fix_group_id]["y"].append(rp.y)
        fix_groups[rp.fix_group_id]["timestamp"].append(int(rp.timestamp))

    rf_inputs = list(map(lambda fix_group: {
        "x": np.median(fix_group['x']),
        "y": np.median(fix_group['y']),
        "timestamp": np.max(fix_group['timestamp']),
        "duration": np.max(fix_group['timestamp']) - np.min(fix_group['timestamp'])
    }, fix_groups.values()))

    rfs = [RawFixation(rf_input) for rf_input in rf_inputs]
    return rfs


def get_rf(rps):
    """
    Description:

    :param rps: List<RawGazePoint> Raw Gaze Point
    :return: List<RawFixation> Raw Fixation
    """
    # Gap Fill In: 현재 모든 데이터에서 좌표가 다 있는 상태라서 굳이 구현하지 않아도 될듯
    # rps = gap_fill_in(rps)

    rps = noise_reduction(rps)
    rps = calculate_velocity(rps)
    rps = absolute_threshold(rps)
    rps = merge_adj_fixation(rps)
    rfs = discard_short_fixation(rps)
    return rfs


def run(rps):
    """
    :param rps: List<RawGazePoint> Raw gaze Point
    :return: rfs: List<RawFixation> Raw Fixation

    현재 단순히 raw fixation만 작동하게 되어 있는데, 데이터가 들어오는 것에 따라서
    전처리나 후처리 등의 과정이 추가될 수도 있는 상태.
    """

    # Raw Gaze Point --> Raw Fixation
    rfs = get_rf(rps)
    return rfs
