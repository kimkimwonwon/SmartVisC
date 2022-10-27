"""
[Mission 2]
Make line allocation Algorithm!
    - Raw Fixation --> Corrected Fixation

Notation:
    - RF : Raw Fixation
    - CF : Corrected Fixation

참고:
Traditional algorithm:
    - use backwark velocity
"""

from utils.data import CorrectedFixation


def allocate_line(rf, word_aoi):
    """
    :param rf: List<RawFixation> Raw Fixation
    :param word_aoi: List<WordAoi> wordAOI
    :return: List<CorrectedFixation> Corrected Fixation
    """
    # TODO: 여기에 RF를 CF로 반환하는 코드 구현하기
    cfs = []

    default = {
        "timestamp": 0,
        "line": 0,
        "order": 0,
        "x": 0,
        "y": 0
    }
    cf = CorrectedFixation(default)
    cfs.append(cf)
    return cfs


def run(rf, word_aoi):
    cf = allocate_line(rf, word_aoi)
    return cf
