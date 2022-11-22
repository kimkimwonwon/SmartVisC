"""
[Mission 3]
Make brand-new metric so that metric value fits well with the traditional estimated value

배정 알고리즘 성능에 영향을 안 받는 지표
    [O] Average Number of Fixations
    [O] Average Fixation Duration
    [0] Fixation Qualitative Score: RP와 RF간 평균 거리로 benchmark는 확인할 수 없다.
    [O] Fixation per Word

배정 알고리즘 성능에 영향을 많이 받는 지표
    - Saccadic Length
    - Regression Ratio
    - Reading Speed

"""
import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import defaultdict
from tqdm import tqdm

import env


def get_ivt_metric_list():
    return [
        "Average Number of Fixations",
        "Average Fixation Duration",
        "Fixation Qualitative Score"
    ]


def get_lineAllo_metric_list():
    return [
        "Fixation per Word"
    ]


def _get_distance(coors):
    """
    coors: [[x1, y1, x2, y2], ...]
    """
    coors = np.array(coors)
    distance = np.power(coors[:, 0] - coors[:, 2], 2) + np.power(coors[:, 1] - coors[:, 3], 2)
    distance = np.sqrt(distance).mean()
    return distance


def avg_rf_num(rps, rfs):
    return len(rfs)/len(rps)


def avg_rf_dur(rfs):
    return np.mean([rf.duration for rf in rfs])


def FQIS(rps, rfs):
    """
    Benchmark의 Raw Gaze Point와 Raw Fixation 간의 mapping을 알 수 없어서 benchmark는 사용못할듯
    """
    coors = [
        [rp.x, rp.y, rfs[rp.fix_group_id].x, rfs[rp.fix_group_id].y]
        for rp in rps
    ]
    distance = _get_distance(coors)
    return distance


def get_ivt_dashboard(rps, point_cur, point_bm):
    db = pd.DataFrame(columns=["Current", 'Benchmark'], index=get_ivt_metric_list())

    num = [avg_rf_num(rps, point_cur), avg_rf_num(rps, point_bm)]
    db.iloc[0] = num

    duration = [avg_rf_dur(point_cur), avg_rf_dur(point_bm)]
    db.iloc[1] = duration

    quality_score = [FQIS(rps, point_cur), -1]
    db.iloc[2] = quality_score

    print(tabulate(db, headers='keys', tablefmt='psql', showindex=True,  floatfmt=".3f"))
    return db


def fpw(word_aoi, cfs):
    """
    아직 단어 배정이 안되어서 정확한 것은 아님
    """
    count = defaultdict(int)
    for cf in cfs:
        k = f"line{cf.line}_order{cf.order}"
        count[k] += 1

    if env.SHOW_ALL_PLOTS:
        df = pd.DataFrame.from_dict(count, orient='index')
        return np.mean(list(count.values())), df
    else:
        return np.mean(list(count.values()))


def get_lineAllo_dashboard(rps, word_aoi, point_cur, point_bm):
    db = pd.DataFrame(columns=["Current", 'Benchmark'], index=get_lineAllo_metric_list())

    fpw_cur, fpw_cur_df = fpw(word_aoi, point_cur)
    fpw_bm, fpw_bm_df = fpw(word_aoi, point_bm)
    fx_per_word = [fpw_cur, fpw_bm]
    db.iloc[0] = fx_per_word

    if env.SHOW_ALL_PLOTS:
        fpw_df = fpw_cur_df.join(fpw_bm_df, lsuffix='Current', rsuffix='BM', how="outer")
        print(tabulate(fpw_df, headers='keys', tablefmt='psql', showindex=True, floatfmt=".3f"))
    print(tabulate(db, headers='keys', tablefmt='psql', showindex=True,  floatfmt=".3f"))
    return db


def export_excel(cfs, is_save=False):
    raw = []
    for cf in cfs:
        raw.append(cf.__dict__)
    df = pd.DataFrame(raw)
    print(df.head(10))
    df.to_excel("data/result_line.xlsx")


def export_all(handler):
    res = pd.DataFrame([])
    prog = tqdm(enumerate(handler.data))
    for i, dat in prog:
        tmp_cfs = dat.correctedFixationList
        raw = []
        for cf in tmp_cfs:
            raw.append(cf.__dict__)
        df = pd.DataFrame(raw)
        df['ID'] = i
        res = pd.concat((res, df), axis=0)
        prog.set_description(f"{i} of {len(handler.data)} Excel exporting")
    return res


def eval_metric():
    """
    Description: compare traditional estimated value with our metric algorithm

    :return:
    """
    return None
