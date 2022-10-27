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
import numpy as np
from utils.data import CorrectedFixation
from collections import defaultdict


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
    xs = np.array([i.x for i in rfs])

    delta_x = np.lib.stride_tricks.sliding_window_view(xs, 2, writeable=True)
    delta_x = delta_x[:, 1] - delta_x[:, 0]

    # Line allocation: Backward
    line_idx = np.zeros_like(delta_x)
    line_idx[delta_x < -600] = 1
    line_idx = line_idx.cumsum().astype(int)

    # 값 업데이트
    line_idx = np.append(line_idx, line_idx[-1])
    for rf, group_id in zip(rfs, line_idx):
        setattr(rf, "line_group_id", group_id)

    line_groups = defaultdict(lambda: defaultdict(list))
    for rf in rfs:
        line_groups[rf.line_group_id]["y"].append(rf.y)
    line_groups = list(map(lambda v: np.median(v['y']), line_groups.values()))

    cfs = []
    line_count = 1
    order_count = 0
    for i, (rf, line_id) in enumerate(zip(rfs, line_idx)):
        if line_id != line_id:
            order = order_count
        else:
            order_count = 0
            order = order_count
            line_count += 1

        cf_input = {
            "timestamp": rf.timestamp,
            "line": line_id,
            "order": order,
            "x": rf.x,
            "y": line_groups[line_id]
        }
        cfs.append(CorrectedFixation(cf_input))
    return cfs


def run(rf, word_aoi):
    cf = allocate_line(rf, word_aoi)
    return cf
