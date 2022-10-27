"""
[Mission 2]
Make line allocation Algorithm!
    - Raw Fixation --> Corrected Fixation

Notation:
    - RF : Raw Fixation
    - CF : Corrected Fixation

참고:
Traditional algorithm:
    - [O] use backwark velocity
    - [X] attach
    - [X] chain
    - [X] cluster
    - [X] compare
    - [X] merge
    - [X] regress
    - [X] segmet
    - [X] split
    - [X] stretch
    - [X] warp
"""

from utils.data import CorrectedFixation


def rf_to_cf(rf):
    config = {
        "timestamp": 0,
        "line": 0,
        "order": 0,
        "x": 0,
        "y": 0
    }
    cf = CorrectedFixation(config)
    return cf


def allocate_line(rfs, word_aoi):
    """
    :param rfs: List<RawFixation> Raw Fixation
    :param word_aoi: List<WordAoi> wordAOI
    :return: List<CorrectedFixation> Corrected Fixation
    """
    # TODO: 기존의 방식이나 새롭게 고안한 방식 사용하기
    rfs = rfs

    cfs = [rf_to_cf(rf) for rf in rfs]
    return cfs


def run(rf, word_aoi):
    cf = allocate_line(rf, word_aoi)
    return cf
