"""
[Mission 1]
Based on the traditional(or conventional) iVT filter process, customize it!
    - Raw Gaze Point --> Raw Fixation


Notation:
    - RP: Raw gaze Point
    - RF : Raw Fixation
"""

from utils.data import RawFixation
import scipy
import numpy as np
import env
import params
from collections import defaultdict
from utils.visual import show_line_plot


def get_xs(obj: list):
    return np.array([i.x for i in obj])


def get_ys(obj: list):
    return np.array([i.y for i in obj])


def gap_fill_in(rps):
    # TBD: Gap fill in 구현
    return rps


def noise_reduction(rps):
    xs = get_xs(rps)
    ys = get_ys(rps)

    # 방법 1: avg
    window_size = 3
    window = np.ones(window_size)/window_size
    xs_new = scipy.signal.convolve(xs, window, mode="same")
    ys_new = scipy.signal.convolve(ys, window, mode="same")
    for rp, x, y in zip(rps, xs_new, ys_new):
        rp.x = x
        rp.y = y
    return rps


def calculate_velocity(rps):
    xs = get_xs(rps)
    ys = get_ys(rps)

    delta_x = np.lib.stride_tricks.sliding_window_view(xs, 2, writeable=True)
    delta_x = delta_x[:, 1] - delta_x[:, 0]
    delta_y = np.lib.stride_tricks.sliding_window_view(ys, 2, writeable=True)
    delta_y = delta_y[:, 1] - delta_y[:, 0]

    # NOTE: 그런데, 여기서 속력이 아니라 속도를 구한다면..?!
    speeds = np.abs(delta_y / delta_x)
    # NA 값 처리
    speeds = np.append(speeds, [0])

    for rp, speed in zip(rps, speeds):
        setattr(rp, "speed", speed)

    if env.show_all_plots:
        show_line_plot(speeds)
    return rps


def absolute_threshold(rps):
    for rp in rps:
        if rp.speed > params.ivt_threshold:
            setattr(rp, "label", "saccade")
        else:
            setattr(rp, "label", "fixation")
    return rps


def merge_adj_fixation(rps):
    # Fixation 사이 Max time: 지금은 죄다 33, 34라서 문제임
    times = np.array([int(i.timestamp) for i in rps])
    delta_t = np.lib.stride_tricks.sliding_window_view(times, 2, writeable=True)
    delta_t = delta_t[:, 1] - delta_t[:, 0]
    fix_group_id = (delta_t > params.max_time_between_fixations).astype(int).cumsum()

    # NA 값 처리
    fix_group_id = np.append(fix_group_id, fix_group_id[-1])
    for rp, group_id in zip(rps, fix_group_id):
        setattr(rp, "fix_group_id", group_id)
    return rps


def discard_short_fixation(rps):
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


def rf_to_rp(rp):
    config = {
        "duration": 0,
        "timestamp": rp.timestamp,
        "x": rp.x,
        "y": rp.y
    }
    rf = RawFixation(config)
    return rf


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
    Description:

    :param rps: List<RawGazePoint> Raw gaze Point
    :return: rfs: List<RawFixation> Raw Fixation
    """

    # Raw Gaze Point --> Raw Fixation
    rfs = get_rf(rps)
    return rfs
