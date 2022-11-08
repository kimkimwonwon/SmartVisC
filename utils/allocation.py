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
import params
import env


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


def classify_backward(rfs: list):
    xs = np.array([rf.x for rf in rfs])
    delta_xs = xs[:-1] - xs[1:]
    delta_xs = np.concatenate((delta_xs, [0]))

    bward_thr = params.backward_threshold
    is_bwards = delta_xs < bward_thr

    fr_count = 0
    seg_id = 0
    for i, (rf, is_bward) in enumerate(zip(rfs, is_bwards)):
        rf.is_backward = is_bward
        if ~ is_bward:
            rf.ftype = "Forward Reading"
            fr_count += 1
        else:
            if i == 0:
                pass
            seg_id += 1
        rf.segment_id = seg_id
    if env.LOG_ALL:
        fr_type_count = len([i for i, rf in enumerate(rfs) if rf.ftype == "Forward Reading"])
        assert fr_count == fr_type_count, "Forward Reading 배정이 잘못되었습니다!"
        print(f"Backward Number : {len(rfs)-fr_count}/{len(rfs)}")
    return rfs


def flatten_segment(rfs: list):
    segment_group = defaultdict(list)
    for rf in rfs:
        segment_group[rf.segment_id].append(rf)

    for group_id, group in segment_group:
        y = [rf.y for rf in group]
        for rf in group:
            rf.y = np.median(y)
    return rfs


def allocate_line(rfs, word_aoi):
    # Parameter : Backward Threshold
    """
    :param rfs: List<RawFixation> Raw Fixation
    :param word_aoi: List<WordAoi> wordAOI
    :return: List<CorrectedFixation> Corrected Fixation

    RawFixation들의 x축의 변화량을 보고 기준치(Backward Threshold)를 넘은 경우에
    다음 id를 할당하는 방식으로 구현되어 있습니다.

    아직 WordAoi를 이용해서 단어별 할당하는 코드는 구현되어 있지 않습니다.
    그리고 아래의 코드는 단순히 구현한거라 이제 제대로 구현해보면 되는 내용입니다...!

    해당 과정에서 Corrected Fixation을 만들어야 하는데 필요한 값은
    (x, y, timestamp, line, order)이며 line은 몇번째 줄인지, order는 그 line 내에서 몇번째 fixation인지
    알려주는 것입니다.
    """
    xs = np.array([i.x for i in rfs])
    backward_threshold = params.backward_threshold

    delta_x = np.lib.stride_tricks.sliding_window_view(xs, 2, writeable=True)
    delta_x = delta_x[:, 1] - delta_x[:, 0]

    # Line allocation: Backward
    # 기준치를 넘은 경우에 1씩 라벨링 하고, 누적합을 구하게 되면, step function처럼 다음 값들이 동일한 id를 가지게 되어서
    # grouping이 가능하게 됩니다!
    line_idx = np.zeros_like(delta_x)
    line_idx[delta_x < backward_threshold] = 1

    line_idx = line_idx.cumsum().astype(int)
    if line_idx[0] == 0:
        line_idx += 1

    # 몇번재 라인인지 각 Raw Fixation에 라벨링
    line_idx = np.append(line_idx, line_idx[-1])
    for rf, group_id in zip(rfs, line_idx):
        setattr(rf, "line_group_id", group_id)

    # line 라벨에 따라서 동일한 line_id를 가지는 경우에는 y축 값이 그 group의 median을 가지도록.
    line_groups = defaultdict(lambda: defaultdict(list))
    for rf in rfs:
        line_groups[rf.line_group_id]["y"].append(rf.y)
    line_groups = list(map(lambda v: np.median(v['y']), line_groups.values()))

    # 각 줄에서 순서대로 order값을 할당해주고 줄이 바뀌는 경우에는 order는 0이 되도록 합니다.
    # 줄이 바뀐지 알기 위해서 line_count 변수를 이용해서 안바뀌면 다음 order 값을, 바뀐 경우에는 order를 0부터 다시 시작하도록 합니다.
    cfs = []
    line_count = 1
    order_count = 1
    for i, (rf, line_id) in enumerate(zip(rfs, line_idx)):
        if line_count == line_id:
            order_count += 1
            order = order_count
        else:
            order_count = 0
            order = order_count
            line_count += 1
        try:
            cf_input = {
                "timestamp": rf.timestamp,
                "line": line_id,
                "order": order,
                "x": rf.x,
                "y": line_groups[line_id-1]
            }
        except :
            print()

        cfs.append(CorrectedFixation(cf_input))
    return cfs


def run(rfs, word_aoi):
    # 전처리(TBD)

    # Raw Fixation --> Corrected Fixation
    rfs = classify_backward(rfs)
    rfs = flatten_segment(rfs)
    cf = allocate_line(rfs, word_aoi)

    # 후처리(TBD)

    return cf
