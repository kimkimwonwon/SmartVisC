"""
[Mission 1]
Based on the traditional(or conventional) iVT filter process, customize it!
    - Raw Gaze Point --> Raw Fixation


Notation:
    - RP: Raw gaze Point
    - RF : Raw Fixation
"""

from utils.data import RawFixation


def get_rf(rp):
    """
    Description:

    :param rp: List<RawGazePoint> Raw Gaze Point
    :return: List<RawFixation> Raw Fixation
    """
    # TODO: 여기에 RP를 RF로 반환하는 코드 구현하기
    rfs = []

    default = {
        "duration": 0,
        "timestamp": 0,
        "x": 0,
        "y": 0
    }
    rf = RawFixation(default)
    rfs.append(rf)
    return rfs


def run(rp):
    """
    Description:

    :param rp: Raw gaze Point
    :return: rf: Raw Fixation
    """

    # Raw Gaze Point --> Raw Fixation
    rf = get_rf(rp)
    return rf
