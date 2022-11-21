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
import utils.const as const
from itertools import combinations


def get_time(obj: list):
    return np.array([i.timestamp for i in obj])


def get_xs(obj: list):
    return np.array([i.x for i in obj])


def get_ys(obj: list):
    return np.array([i.y for i in obj])


def get_coors(obj: list, include_time=False):
    res = np.vstack((get_xs(obj), get_ys(obj))).T
    if include_time:
        res = np.hstack((res, get_time(obj)[:, np.newaxis]))
    return res


def get_edge(rfs: list, return_last=False):
    screen_width = const.screen_width
    backward_ratio = params.backward_ratio
    line_min_fix_num = params.line_min_fix_num

    coors = get_coors(rfs)
    delta = coors[2:] - coors[:-2]
    delta = np.concatenate((np.zeros((2, 2)), delta))

    is_backward = delta[:, 0] < -screen_width * backward_ratio
    seg_id = np.cumsum(is_backward)

    big_step_seg = defaultdict(int)
    for i in seg_id:
        big_step_seg[i] += 1

    first_line_id = -1
    for k, v in big_step_seg.items():
        if v > line_min_fix_num:
            first_line_id = k
            break
    if first_line_id == -1:
        first_line_id = 1
    first_line = coors[seg_id <= first_line_id]

    if not return_last:
        return first_line
    else:
        last_line_id = -1
        for k, v in list(big_step_seg.items())[::-1]:
            if v > line_min_fix_num:
                last_line_id = k
                break
        if last_line_id == -1:
            last_line_id = list(big_step_seg.keys())[-1]
        last_line = coors[seg_id >= last_line_id]
        return first_line, last_line


# ver.1.1: scale 전 처리
def rm_noise(rfs):
    rm_idx = []
    # Step 1: 첫번째 마지막줄 찾기
    last_jump_y = params.last_jump_y

    first_line, last_line = get_edge(rfs, return_last=True)

    # Step 2: 진짜 첫번째줄부터만 남기기
    start_idx = np.argmin(first_line[:, 0])
    for i in range(start_idx):
        rm_idx.append(i)
        rfs[i].ignore = True

    # Step 3: 마지막 줄 남기기
    delta = last_line[2:] - last_line[:-2]
    delta = np.concatenate((delta, np.zeros((2, 2))))
    last_point_idx = np.argwhere(delta[:, 1] > last_jump_y)[0, 0]

    for i in range(len(rfs)-len(last_line)+last_point_idx+1, len(rfs)):
        rm_idx.append(i)
        rfs[i].ignore = True
    rfs = rfs[start_idx:]
    rfs = rfs[:len(rfs)-len(last_line)+last_point_idx+1]
    return rfs


def get_transform(rfs, word_aois):
    """
    우선 위에 이상한 점들이 다 사라졌다고 가정했을 때 시작하는 gaze point가 좌상단에 있을 때라고 가정
    """
    # Step 1: 첫번째 줄에 해당하는 점들을 찾기
    first_line, last_line = get_edge(rfs, return_last=True)

    # Step 2: 시작점과 끝점 찾기
    def get_edge_points(line):
        start_idx = np.argmin(line[:, 0])
        dist = pairwise_distances(line)
        end_idx = np.argmax(dist[start_idx])
        return line[[start_idx, end_idx]]

    fl_points = get_edge_points(first_line)

    # Step 3: 시작 단어와 끝 단어 찾기
    word_lines = defaultdict(list)
    for word in word_aois:
        word_lines[word.line].append(word)
    word_lines_len = list(map(lambda x: len(x), word_lines.values()))
    first_line_id = 0 if word_lines_len[0] != 1 else 1
    end_line_id = list(word_lines.keys())[-1] if word_lines_len[-1] != 1 else list(word_lines.keys())[-2]

    first_line_word = [word for word in word_aois if word.line == first_line_id]

    def get_word_points(line):
        start_word = line[0]
        end_word = line[-1]
        return np.array([[start_word.wordBox.x, start_word.wordBox.y], [end_word.wordBox.x, end_word.wordBox.y]])

    fl_wpoints = get_word_points(first_line_word)

    # Step 4: 변환 transform 찾기
    scale = np.linalg.norm(fl_wpoints[0]-fl_wpoints[1])/np.linalg.norm(fl_points[0]-fl_points[1])
    delta = fl_points[1]-fl_points[0]
    slope = np.abs(delta[1]/(delta[0]+1e-6))
    theta = - np.arctan(slope)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Step 5: updates
    for rf in rfs:
        coors = np.array([[rf.x], [rf.y]])
        new_point = rot @ coors
        new_point = new_point[:, 0]
        setattr(rf, "x", new_point[0]*scale)
        setattr(rf, "y", new_point[1]*scale)
    return rfs


# ver.1.2: 기준점 조정
def get_offset(rfs, word_aois):
    coors = get_coors(rfs)
    text_x = [i.wordBox.x for i in word_aois]
    text_y = [i.wordBox.y for i in word_aois]

    for rf in rfs:
        rf.x -= np.min(coors[:, 0]) - min(text_x)
        rf.y -= np.min(coors[:, 1]) - min(text_y)
    return rfs


# get index per line
def get_idx(word_aois):
    idx = defaultdict(list)
    
    for j in range(word_aois[-1].line+1):
        for i in range (len(word_aois)):
            if word_aois[i].line == j:
                idx[j].append(i)
    return idx

# Get nC2 of line
def get_idx_combo(word_aois):
    lsts = [i for i in range(word_aois[-1].line+1)]

    combo = list(combinations(lsts, 2))
    return combo

# Get mean of word_cnt
def word_cnt_mean(word_aoi):
    word_cnt_lst = [word_aoi[i].word_cnt for i in range(len(word_aoi))]
    cnt_mean = sum(word_cnt_lst)/len(word_cnt_lst)
    return cnt_mean

# When there are n words in a sentence, 
# returns the length of n-1 words except the last word and the number of n-1 gaps
def letter_and_gap(word_aois, idx1, idx2):
    # line 0: 단어 개수(마지막 단어 제외), 띄어쓰기 개수
    word_cnt_1 = sum([(word_aois[idx1[i]].word_cnt)-1 for i in range(len(idx1)-1)]) 
    space_cnt_1 = len(idx1)-2
    # line 1: 단어 개수(마지막 단어 제외), 띄어쓰기 개수
    word_cnt_2 = sum([(word_aois[idx2[i]].word_cnt)-1 for i in range(len(idx2)-1)])
    space_cnt_2 = len(idx2)-2
    # line 길이
    len_1 = word_aois[idx1[-1]].wordBox.x-word_aois[idx1[0]].wordBox.x
    len_2 = word_aois[idx2[-1]].wordBox.x-word_aois[idx2[0]].wordBox.x
            
    let = np.array([[word_cnt_1, space_cnt_1],[word_cnt_2, space_cnt_2]])
    gap = np.array([len_1, len_2])
    
    return let, gap

# Get the length of a single word and the length of a gap
# backward_threshold = mean of word_cnt * mean of single word lengths + mean of gap lengths
# default: return backward_threshold
# use_gap == True: return the mean of gaps
def get_seg_thr(word_aois, use_params = False, use_thr = True, use_gap = False):
    if use_params == True:
        return params.backward_threshold
    else:
        let = []
        gap = []
        idx = get_idx(word_aois)
        cnt_mean = word_cnt_mean(word_aois)

        for i,j in get_idx_combo(word_aois):
            result = letter_and_gap(word_aois, idx[i], idx[j])
            if np.linalg.det(result[0]) != 0 :
                let.append(result[0])
                gap.append(result[1])
            else : pass
        thr = np.linalg.solve(let, gap)
        thr = np.where((thr > 0) & (thr < const.font_size), thr, 0)
        gap_threshold = np.mean(thr[:,1])
        backward_threshold = -(cnt_mean*(np.mean(thr[:,0])) + np.mean(thr[:,1]))
        
        if use_gap == True:
            return gap_threshold
        else:
            return backward_threshold
        # assert False, "아직 함수를 구현하지 않았습니다!"

# ver.1.1: backward movement 감지
def classify_backward(rfs: list, word_aois):
    xs = np.array([rf.x for rf in rfs])
    delta_xs = xs[:-1] - xs[1:]
    delta_xs = np.concatenate((delta_xs, [0]))

    # NOTE: hyper parameter
    bward_thr = get_seg_thr(word_aois)
    is_bwards = delta_xs < bward_thr

    fr_count = 0
    seg_id = 0
    for i, (rf, is_bward) in enumerate(zip(rfs, is_bwards)):
        rf.is_backward = is_bward
        if ~ is_bward:
            rf.ftype = "Forward Reading"
            fr_count += 1
        else:
            if i != 0:
                seg_id += 1
        rf.segment_id = seg_id
    if env.LOG_ALL:
        fr_type_count = len([i for i, rf in enumerate(rfs) if rf.ftype == "Forward Reading"])
        assert fr_count == fr_type_count, "Forward Reading 배정이 잘못되었습니다!"
        print(f"Backward Number : {len(rfs)-fr_count}/{len(rfs)}")
    return rfs


# # ver.0.1: backward movement 감지 (hyper parameter issue)
# def classify_backward(rfs: list):
#     xs = np.array([rf.x for rf in rfs])
#     delta_xs = xs[:-1] - xs[1:]
#     delta_xs = np.concatenate((delta_xs, [0]))

#     # NOTE: hyper parameter
#     bward_thr = get_seg_thr()
#     is_bwards = delta_xs < bward_thr

#     fr_count = 0
#     seg_id = 0
#     for i, (rf, is_bward) in enumerate(zip(rfs, is_bwards)):
#         rf.is_backward = is_bward
#         if ~ is_bward:
#             rf.ftype = "Forward Reading"
#             fr_count += 1
#         else:
#             if i != 0:
#                 seg_id += 1
#         rf.segment_id = seg_id
#     if env.LOG_ALL:
#         fr_type_count = len([i for i, rf in enumerate(rfs) if rf.ftype == "Forward Reading"])
#         assert fr_count == fr_type_count, "Forward Reading 배정이 잘못되었습니다!"
#         print(f"Backward Number : {len(rfs)-fr_count}/{len(rfs)}")
#     return rfs


def rm_peak(rfs: list, word_aois):
    rm_idx = []
    peak_mv = params.peak_mv
    peak_pad = params.peak_pad

    # Step 1: 텍스트의 영역 + pad의 공간 밖의 점들은 무시
    wxs = [i.wordBox.x for i in word_aois]
    wys = [i.wordBox.y for i in word_aois]
    wx_min, wx_max, wy_min, wy_max = min(wxs), max(wxs), min(wys), max(wys)
    for i, rf in enumerate(rfs):
        if (rf.x < wx_min-peak_pad) or (rf.x > wx_max+peak_pad) or (rf.y < wy_min-peak_pad) or (rf.y > wy_max+peak_pad):
            rm_idx.append(i)
            rf.ignore = True

    # Step 2: 중간에 갑자기 움직이는 애들. 이러한 경우 배정만 안할 뿐 기록은 가지고 있어야 한다!
    coors = get_coors(rfs)
    delta = coors[1:] - coors[:-1]
    delta = np.concatenate((np.zeros((1, 2)), delta))
    for i, (delta_i, rf) in enumerate(zip(delta, rfs)):
        if (i != len(rfs)-1) and (np.abs(delta_i[1]) > peak_mv) and (np.abs(delta[i+1][1]) > peak_mv):
            rm_idx.append(i)
    res = [rf for i, rf in enumerate(rfs) if i not in set(rm_idx)]
    return res


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

        # line_x *= (np.max(segment_x)-np.min(segment_x)) / (np.max(line_x) - np.min(line_x))
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
    cfs[0].ftype = "Forward Reading"
    for rf in rfs[1:]:
        past = cfs[-1]
        if past.line == rf.line_id and past.order == rf.order_id:
            past.duration += rf.timestamp - past.timestamp
        else:
            cfs.append(CorrectedFixation(_get_inputs(rf, word_aois)))

    for cf_past, cf_cur in zip(cfs[:-1], cfs[1:]):
        if cf_past.line == cf_cur.line:
            if cf_cur.order < cf_past.order:
                cf_cur.ftype = "Inline Regression"
            else:
                cf_cur.ftype = "Forward Reading"
        else:
            if cf_cur.line - cf_past.line == 1:
                cf_cur.ftype = "Line Change"
            elif cf_cur.line - cf_past.line > 1:
                cf_cur.ftype = "Scan"
            else:
                cf_cur.ftype = "Between Line Regression"
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
    rfs = rm_noise(rfs)
    rfs = get_transform(rfs, word_aois)
    rfs = rm_peak(rfs, word_aois)
    rfs = get_offset(rfs, word_aois)
    # rfs = flatten_segment(rfs)

    # 작은 segment로 나누기
    rfs = classify_backward(rfs, word_aois)

    # 후처리(TBD)
    rfs = allocate_line_id(rfs, word_aois)
    rfs = allocate_order_id(rfs, word_aois)
    cfs = to_CorrectedFixation(rfs, word_aois)

    if env.LOG_ALL:
        ftype_status = defaultdict(int)
        for cf in cfs:
            assert cf.ftype is not None, "Missing Fixation type exists"
            ftype_status[cf.ftype] += 1
        print(ftype_status)
    return cfs
