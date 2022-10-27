"""
[Mission 1]
Based on the traditional(or conventional) iVT filter process, customize it!
    - Raw Gaze Point --> Raw Fixation


Notation:
    - RP: Raw gaze Point
    - RF : Raw Fixation
"""

from utils.data import RawFixation


def gap_fill_in(rps):
    # TODO: Gap fill in 구현
    return rps


def noise_reduction(rps):
    # TODO: noise_reduction 구현
    return rps


def calculate_velocity(rps):
    # TODO: calculate_velocity 구현
    return rps


def absolute_threshold(rps):
    # TODO: absolute_threshold 구현
    return rps


def merge_adj_fixation(rps):
    # TODO: merge_adjacent_fixation 구현
    return rps


def discard_short_fixation(rps):
    # TODO: discard_short_fixation 구현
    return rps


def rf_to_rp(rp):
    config = {
        "duration": 0,
        "timestamp": 0,
        "x": 0,
        "y": 0
    }
    rf = RawFixation(config)
    return rf


def get_rf(rps):
    """
    Description:

    :param rps: List<RawGazePoint> Raw Gaze Point
    :return: List<RawFixation> Raw Fixation
    """
    # TODO: 여기에 RP를 RF로 반환하는 코드 구현하기
    rps = gap_fill_in(rps)
    rps = noise_reduction(rps)
    rps = calculate_velocity(rps)
    rps = absolute_threshold(rps)
    rps = merge_adj_fixation(rps)
    rps = discard_short_fixation(rps)

    rfs = [rf_to_rp(rp) for rp in rps]
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
