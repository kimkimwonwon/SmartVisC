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
from sklearn.metrics import pairwise_distances


# ver.0.1: 전체 크기  조정
def scaling(rfs):
    coors = [[rf.x, rf.y] for rf in rfs]
    coors = np.array(coors)
    x_min, y_min = np.min(coors, axis=0)

    for rf in rfs:
        rf.x -= x_min
        rf.y -= y_min
    return rfs


# ver.0.1: backward movement 감지 (hyper parameter issue)
def classify_backward(rfs: list):
    xs = np.array([rf.x for rf in rfs])
    delta_xs = xs[:-1] - xs[1:]
    delta_xs = np.concatenate((delta_xs, [0]))

    # NOTE: hyper parameter
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


# ver.0.1: 점들 높이 맞추기
def flatten_segment(rfs: list):
    segment_group = defaultdict(list)
    for rf in rfs:
        segment_group[rf.segment_id].append(rf)

    for group_id, group in segment_group.items():
        y = [rf.y for rf in group]
        for rf in group:
            rf.y = np.median(y)
    return rfs


# ver.0.1: 줄 배정
def allocate_line_id(rfs, word_aois):
    line_ys = [None] * (word_aois[-1].line+1)
    for word_aoi in word_aois:
        line_ys[word_aoi.line] = word_aoi.wordBox.y
    line_ys = np.array(line_ys)[:, np.newaxis]

    segment_ys = [None] * (rfs[-1].segment_id+1)
    for rf in rfs:
        segment_ys[rf.segment_id] = rf.y
    segment_ys = np.array(segment_ys)[:, np.newaxis]

    distances = pairwise_distances(segment_ys, line_ys)
    line_idx = np.argmin(distances, axis=1)

    for rf in rfs:
        rf.line_id = line_idx[rf.segment_id]
    return rfs


# ver.0.1 : 단어 배정
def allocate_order_id(rfs, word_aois):
    line_group = defaultdict(list)
    for word_aoi in word_aois:
        line_group[word_aoi.line].append(word_aoi)

    segment_group = defaultdict(list)
    for rf in rfs:
        segment_group[rf.segment_id].append(rf)

    for segment_group_id, segment in segment_group.items():
        line_id = segment[0].line_id
        line = line_group[line_id]
        line_x = np.array([word.wordBox.x for word in line])[:, np.newaxis]
        segment_x = np.array([point.x for point in segment])[:, np.newaxis]

        distances = pairwise_distances(segment_x, line_x)
        word_idx = np.argmin(distances, axis=1)
        for rf, word_id in zip(segment, word_idx):
            rf.order_id = word_id

    if env.LOG_ALL:
        wrong_cnt = 0
        total_cnt = 0
        for segment in segment_group.values():
            word_idx = [word.order_id for word in segment]
            word_idx = np.array(word_idx)
            total_cnt += len(word_idx)
            wrong_cnt += ((word_idx[1:]-word_idx[:-1]) < 0).sum()

        print(f"잘못된 단어(order id) 배정 횟수: {wrong_cnt}/{total_cnt}")
    return rfs


def _get_inputs(rf, word_aois):
    for word_aoi in word_aois:
        if rf.line_id == word_aoi.line and rf.order_id == word_aoi.order:
            word = word_aoi

    cf_input = {
        "duration": 30,
        "timestamp": rf.timestamp,
        "line": rf.line_id,
        "order": rf.order_id,
        "x": word.wordBox.x,
        "y": word.wordBox.y
    }
    return cf_input


# ver.0.1: CF로 만들기
def to_CorrectedFixation(rfs, word_aois):
    cfs = [CorrectedFixation(_get_inputs(rfs[0], word_aois))]
    for rf in rfs[1:]:
        past = cfs[-1]
        if past.line == rf.line_id and past.order == rf.order_id:
            past.duration += rf.timestamp - past.timestamp
        else:
            cfs.append(CorrectedFixation(_get_inputs(rf, word_aois)))
    return cfs


# Deprecated
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


def run(rfs, word_aois):
    # 전처리(TBD)

    # Raw Fixation --> Corrected Fixation
    rfs = classify_backward(rfs)
    rfs = scaling(rfs)
    rfs = flatten_segment(rfs)

    rfs = allocate_line_id(rfs, word_aois)
    rfs = allocate_order_id(rfs, word_aois)
    cfs = to_CorrectedFixation(rfs, word_aois)
    # 후처리(TBD)

    return cfs
