"""
[Mission 1]
Based on the traditional(or conventional) iVT filter process, customize it!
    - Raw Gaze Point --> Raw Fixation


Notation:
    - RP: Raw gaze Point
    - RF : Raw Fixation
"""
from utils.data import RawFixation
import scipy.signal as signal
# from scipy import signal
import numpy as np
import pandas as pd
import env
import params
from collections import defaultdict
from utils.visual import show_line_plot, show_line_plot_compare
from sklearn.metrics import pairwise_distances


def get_xs(obj: list):
    return np.array([i.x for i in obj])


def get_ys(obj: list):
    return np.array([i.y for i in obj])


def get_time(obj: list):
    return np.array([i.timestamp for i in obj])


def get_delta(obj, window=2):
    delta = np.lib.stride_tricks.sliding_window_view(obj, window, writeable=True)
    delta = delta[:, -1] - delta[:, 0]
    return delta

def time_correction(rps):

  """
  :param rps: List<RawGazePoint>
  :return: List<RawGazePoint>
  timestamp가 중복인 경우를 찾고, 중복 바로 뒤에 나오는 timestamp과, 중복값들 중 첫 timestamp을 비교
  1. 중복 바로 뒤 timestamp - 중복값들 중 첫 timestamp > 40 
  : 단순 중복이 아닌 경우
  1) 중복 바로 뒤 x,y값이 결측값이 아닌 경우 : 중복값에 대하여 그 자리에 timestamp, x, y를 선형으로 fill-in
  2) 중복 바로 뒤 x,y값이 결측값인 경우 : 중복값에 대하여 그 자리에 timestamp는 선형으로 fill-in하고 x,y는 결측값 처리
  -> 결측값으로 처리됨으로써 뒤에서 gap-fill-in에서 처리될 수 있도록 함.
  2. 중복 바로 뒤 timestamp - 중복값들 중 첫 timestamp <= 40
  : 단순 중복인 경우임. 40보다 차이가 덜 난다는 것은, 그냥 똑같은 값이 두 번 들어간 것이고 시선이 두 개였다고 보기는 어려움.
  : 중복값을 그냥 없애줌.
  """

  time = get_time(rps)
  xs = get_xs(rps)
  ys = get_ys(rps)
  xs_new = xs.copy()
  ys_new = ys.copy()
  time_new = time.copy()
  dup = []
  just_error = []

  for i in range(1,len(time)-1):
    if time[i] == time[i-1]:
      dup.append(i-1)
      dup.append(i)
    elif dup==[]:
      pass
    else:
      dup = set(dup)
      dup = sorted(list(dup))
      time_after = time[dup[-1]+1]
      time_before = time[dup[0]]
      time_distance = time_after-time_before
      
      if time_distance > 40: # 1.중복 바로 뒤 timestamp - 중복값들 중 첫 timestamp > 40 
        x_after = xs[dup[-1]+1] # 중복값 끝난 뒤에 제대로 나오는 정상적인 값
        x_before = xs[dup[0]] # 중복값 처음에 나오는 제대로 된 값
        x_distance = x_after-x_before
        y_after = ys[dup[-1]+1] # 중복값 끝난 뒤에 제대로 나오는 정상적인 값
        y_before = ys[dup[0]] # 중복값 처음에 나오는 제대로 된 값
        y_distance = y_after-x_before

        if ((~np.isnan(x_after)) and (~np.isnan(y_after))): # 1) 중복 바로 뒤 x,y값이 결측값이 아닌 경우
          for j in range(1,len(dup)):
            time_new[dup[j]] = int(time_before + (time_distance/len(dup))*j)
            xs_new[dup[j]] = x_before + (x_distance/len(dup))*j
            ys_new[dup[j]] = y_before + (y_distance/len(dup))*j
        else: # 2) 중복 바로 뒤 x,y값이 결측값인 경우
          for j in range(1,len(dup)):
            time_new[dup[j]] = int(time_before + (time_distance/len(dup))*j)
            # timestamp는 선형 보간
            xs_new[dup[j]] = np.nan
            ys_new[dup[j]] = np.nan
            # xs와 ys 모두 결측으로 처리하여 gapfillin에서 알아서 처리하도록.

      else: # 2. 중복 바로 뒤 timestamp - 중복값들 중 첫 timestamp <= 40
        for duplication in dup[1:]:
          just_error.append(duplication)

      dup = []

  for error in just_error[::-1]:
    # 뒤에서부터 제거해야 안정적으로 index에 맞춰 제거 가능
      xs_new = np.delete(xs_new, error)
      ys_new = np.delete(ys_new, error)
      time_new = np.delete(time_new, error)
      rps.pop(error)

  for rp, x, y, time in zip(rps, xs_new, ys_new,time_new):
    rp.x = x
    rp.y = y
    rp.timestamp = time
  
  return rps

def gap_fill_in(rps):

  """
  :param rps: List<RawGazePoint>, max_gap_length, max_fill_length, max_x_distance, max_y_distance
  :return: List<RawGazePoint>
  max_gap_length : 선형으로 interpolate 할 수 있는 gap의 최소 길이. 논문에서 '보간 뒤에도 최대한으로 존재할 수 있는 gap의 길이'로 정의되어 변수명을 이렇게 함.
  max_fill_length :선형으로 interpolate 할 수 있는 gap의 최대 길이. 너무 gap이 크면 보간하는 것이 의미가 없기 때문
  max_x_distance : 선형으로 interpolate 하기 위한 조건, gap에서의 x 좌표 차가 이보다 작아야 함 
  max_y_distance : 선형으로 interpolate 하기 위한 조건, gap에서의 y 좌표 차가 이보다 작아야 함 
  1. max_gap_length < gap의 기간 < max_fill_length
  1) x 좌표 차이 < max_x_distance & y 좌표 차이 < max_y_distance : timestamp에 비례하여 선형으로 보간
  2) otherwise : timestamp를 연속적으로 만들기
  2. gap의 기간 < max_gap_length (즉 1,2개씩만 결측이 연속으로 존재하는 경우) : 그냥 gap 자체를 drop ; 굳이 timestamp를 당길 이유가 없음
  3. gap의 기간 > max_fill_length : timestamp를 연속적으로 만들기
  """
  mgl = params.max_gap_length
  mfl = params.max_fill_length
  myd = params.max_y_distance
  mxd = params.max_x_distance
  time = get_time(rps)
  xs = get_xs(rps)
  ys = get_ys(rps)
  xs_new = xs.copy()
  ys_new = ys.copy()
  time_new = time.copy()
  gaps = []
  not_filled = []
  count = 0

  for i in range(len(xs)):
    
    if (np.isnan(xs[i])) or (np.isnan(ys[i])):
      gaps.append(i)
      count=count+1
    elif count==0:
      pass
    else:
      time_after = time[gaps[-1]+1]
      time_before = time[gaps[0]-1]
      time_duration = time_after-time_before
      if (time_duration >= mgl) and (time_duration <= mfl):
        xbefore = xs[gaps[0]-1]
        xafter = xs[gaps[-1]+1]
        ybefore = ys[gaps[0]-1]
        yafter = ys[gaps[-1]+1]
        if (abs(ybefore - yafter) < myd) and (abs(xbefore-xafter) < mxd): 
          for gap in gaps:
            xs_new[gap] = xafter - ((time_after-time[gap])/time_duration)*(xafter-xbefore) ## scaler 정의 및 gap fill in
            ys_new[gap] = yafter - ((time_after-time[gap])/time_duration)*(yafter-ybefore)
        else:
          time_new[gaps[-1]+1:] = time_new[gaps[-1]+1:] - time_duration+33 # gap 이후의 것들의 timestamp 당겨오기
          # 보통 33 가량 timestamp 차이나므로, 당겨온 간격은 33으로 설정.
          # gap에서의 값들은, 뒤에서 삭제될 예정이기에 딱히 고려할 필요 없음. gap보다 뒤의 값들에 대해서만 당겨오기
          for gap in gaps:
            not_filled.append(gap)
      elif(time_duration < mgl):
        for gap in gaps:
          not_filled.append(gap)
      else: #time_duration > mfl
        time_new[gaps[-1]+1:] = time_new[gaps[-1]+1:] - time_duration+33 # gap 이후의 것들의 timestamp 당겨오기
        # 보통 33 가량 timestamp 차이나므로, 당겨온 간격은 33으로 설정.
          # gap에서의 값들은, 뒤에서 삭제될 예정이기에 딱히 고려할 필요 없음. gap보다 뒤의 값들에 대해서만 당겨오기
        for gap in gaps:
          not_filled.append(gap)
      if i != (len(xs)-1): # 마지막에 nan 값이 존재하는 경우를 처리하기 위함.
        count=0
        gaps=[]

  # 위의 for문은, 마지막에 nan 값이 존재하는 경우를 고려하지 못하므로 따로 처리
  if len(xs)!=0:
    if (np.isnan(xs[(len(xs)-1)])) or (np.isnan(ys[(len(xs)-1)])):
      for gap in gaps:
        not_filled.append(gap)

  for gap in not_filled[::-1]: # timestamp 당겨서 gap을 제거하는 경우, 그냥 제거하는 경우 모두에 대해서 없애기
  # 뒤에서부터 제거해야 안정적으로 index에 맞춰 제거 가능
    xs_new = np.delete(xs_new, gap)
    ys_new = np.delete(ys_new, gap)
    time_new = np.delete(time_new, gap)
    rps.pop(gap)


  for rp, x, y, time in zip(rps, xs_new, ys_new,time_new):
    rp.x = x
    rp.y = y
    rp.timestamp = time
  return rps


def noise_reduction(rps):
    # Parameter: Window size for Moving average
    """
    :param rps: List<RawGazePoint>
    :return: List<RawGazePoint>
    Baseline으로 현재 Moving average를 한 값으로 기존의 값을 업데이트 진행
        - 사실 신호에 대해 convolution연산을 하는 것과 비슷한 것
    """
    xs = get_xs(rps)
    ys = get_ys(rps)

    # # 방법 1: avg
    # window_size = params.window_size
    # window = np.ones(window_size)/window_size
    # xs_new = signal.convolve(xs, window, mode="same")
    # ys_new = signal.convolve(ys, window, mode="same")
    # for rp, x, y in zip(rps, xs_new, ys_new):
    #     rp.x = x
    #     rp.y = y

    # # NOTE: ver.0.1: noise reduction 효과에 따라 시계열 그림으로 나타냄.
    # if env.SHOW_ALL_PLOTS:
    #     show_line_plot_compare(xs, xs_new, f"Noise Reduction, X with window size = {window_size}")
    #     show_line_plot_compare(ys, ys_new, f"Noise Reduction, Y with window size = {window_size}")
    #방법 2 : Exponential smoothing
    #t 시점의 actual 관측치 * alpha + t-1 예측값 *( 1-alpha)

    alpha= 0.7 #alpha값은 메트릭을 고려하여 추후 변경 
    xs_new= [0]*len(xs)
    ys_new= [0]*len(ys)
    for idx,i,in enumerate(xs_new):
        if idx==0:
            xs_new[idx]= xs[0]
        else: 
            xs_new[idx] = alpha* (xs[idx])+ (1-alpha)*(xs_new[idx-1]) 
    for idx,i,in enumerate(ys_new):
        if idx==0:
            ys_new[idx]= ys[0]
        else: 
            ys_new[idx] = alpha* (ys[idx])+ (1-alpha)*(ys_new[idx-1]) 

    for rp, x, y in zip(rps, xs_new, ys_new):
        rp.x = x
        rp.y = y

    return rps

    # #방법 3 simple kalman filter
    
    # ''' params :
    # A : 이전값을 통해 추정값을 예측할때 사용되는 행렬 
    # P : 예측한 값에 대한 오차의 공분산
    # Q : 시스템 노이즈  
    # R : 측정값 노이즈 
    # H : 예측값을 측정값으로 변환할때 사용되는 행렬  
    # K : 칼만이득 => 추정치 - 오차공분산 
    # '''

    # # initial value 
    # A= 1
    # H= 1 
    # Q= np.var(xs)
    # R= np.var(xs)
    # # initial estimate 
    # x_0 = xs[0]
    # y_0 = ys[0]
    # P_0 = 1
    # Kx_0 = 1
    # Ky_0 = 1


    # # z meas - >관측값 
    # def kalman_filter(z_meas, x_esti, P):
    # # (1) Prediction.
    #     x_pred = A * x_esti
    #     P_pred = A * P * A + Q

    # # (2) Kalman Gain.
    #     K = P_pred * H / (H * P_pred * H + R)

    # # (3) Estimation.
    #     x_esti = x_pred + K * (z_meas - H * x_pred)

    # # (4) Error Covariance.
    #     P = P_pred - K * H * P_pred

    #     return x_esti, P, K

    
    # xs_new= [0]*len(xs)
    # ys_new= [0]*len(ys)


    # for idx,i in enumerate(xs_new):
    #     if idx==0:
    #         xs_new[idx], P , K= x_0,P_0 ,Kx_0
    #     else:
    #         xs_new[idx],P, K = kalman_filter(xs[idx],xs_new[idx-1],P)

    # for idx, i in enumerate(ys_new):
    #     if idx ==0 :
    #         ys_new[idx],P,K = y_0,P_0,Ky_0
    #     else:
    #         ys_new[idx],P, K = kalman_filter(ys[idx],ys_new[idx-1],P)

    # for rp, x, y in zip(rps, xs_new, ys_new):
    #     rp.x = x
    #     rp.y = y

    # return rps




def calculate_velocity(rps):
    #Input= rps: rps에서 x좌표 y좌표, timestamp 사용
    #Parameter : window length(=2)
    #Output= 기존의 rps에 수정된 velocity 적용, 단 앞 뒤로 비는 부분은 인접 velocity로 적용 


    window_len = params.window_len

    times = get_time(rps)
    xs = get_xs(rps)
    ys = get_ys(rps)

    delta_time = get_delta(times, window_len)
    delta_x = get_delta(xs, window_len)
    delta_y = get_delta(ys, window_len)    
    distances=pow(pow(delta_x,2)+pow(delta_y,2),1/2)
    speeds = np.abs(distances / delta_time)


  
    diff=len(rps)-len(speeds) #2(L-1) window size<<data size란 가정 아래
    first_value=speeds[0]

    for n in range(int(diff/2)):
        speeds = np.insert(speeds,0,first_value)
    speeds = np.append(speeds, [speeds[-1]]*(len(rps)-len(speeds)))

    #window length가 2가 아닐 때를 대비하여 범용성 있게 작성했습니다.
    #window length가 2인 경우, np.append(speeds,speeds[-1])과 동일한 코드가 됩니다.

    # 각 point에 대해 속력값 업데이트
    for rp, speed in zip(rps, speeds):
        setattr(rp, "speed", speed)

    if env.SHOW_ALL_PLOTS:
        show_line_plot(speeds, f"Speed plot with window length = {window_len}")
    return rps


def ivt_classifier(rps):
    # Parameter : Velocity Threshold
    """
    :param rps: List<RawGazePoint>
    :return: List<RawFixation>
    velocity threshold를 넘으면 saccade 아래면 fixation으로 설정
    saccade인 순간 해당 gaze point를 현재 fixation에 넣고, 다음 gaze point부터는 다음 fixation group에 들어가도록 fix_group_id 최신화
    fixation인 순간 해당 gaze point를 현재 fixation에 포함
    """

    velocity_threshold = params.velocity_threshold

    fix_groups = defaultdict(lambda: defaultdict(list))
    fix_group_id = 0

    for rp in rps[:-1]:
        if rp.speed > velocity_threshold:
            setattr(rp, "label", "saccade")
            rp.fix_group_id = fix_group_id
            fix_groups[fix_group_id]["x"].append(rp.x)
            fix_groups[fix_group_id]["y"].append(rp.y)
            fix_groups[fix_group_id]["timestamp"].append(rp.timestamp)
            fix_group_id += 1
        else:
            setattr(rp, "label", "fixation")
            rp.fix_group_id = fix_group_id
            fix_groups[fix_group_id]["x"].append(rp.x)
            fix_groups[fix_group_id]["y"].append(rp.y)
            fix_groups[fix_group_id]["timestamp"].append(rp.timestamp)

    rf_inputs = list(map(lambda fix_group : {
        "x" : np.median(fix_group["x"]),
        "y" : np.median(fix_group["y"]),
        "timestamp" : np.max(fix_group["timestamp"]),
        "duration" : np.max(fix_group["timestamp"]) - np.min(fix_group["timestamp"])
    }, fix_groups.values()))

    rfs = [RawFixation(rf_input) for rf_input in rf_inputs]

    return rfs


def merge_adj_fixation(rps):
    # Parameter: Max time between fixations
    # 추가로 cluster threshold, The Number of neighbors 를 사용
    """
    :param rps: List<RawGazePoint>
    :return: List<RawGazePoint>
    Fixation 사이 Max time: 지금은 죄다 33, 34라서 문제라서 Deprecated된 코드는 사용하지 못하고 있음
    새로 데이터를 받았을 때 확인해봐야 하는 부분.

    Baseline으로 인접하다고 묶인 경우에 대해서 fix_group_id라는 attr를 업데이트 함!
    즉 각 작은 점들마다 합쳐서 하나의 점들로 생각할 경우에는 동일한 fix_group_id를 가지도록

    cluste
    """

    # # Deprecated
    # times = np.array([int(i.timestamp) for i in rps])
    # delta_t = np.lib.stride_tricks.sliding_window_view(times, 2, writeable=True)
    # delta_t = delta_t[:, 1] - delta_t[:, 0]
    # fix_group_id = (delta_t > params.max_time_between_fixations).astype(int).cumsum()

    ###########################################################################
    # Euclidean distance를 기준으로 묶는 과정
    # 다만 어떤 기준으로 묶는지는 임시로 구현해놓은 것

    cluster_thr = params.cluster_thr
    num_neighbor = params.num_neighbor

    xs = get_xs(rps)[:, np.newaxis]
    ys = get_ys(rps)[:, np.newaxis]
    coors = np.concatenate((xs, ys), axis=1)
    distance_mat = pairwise_distances(coors)

    # 임시로 정한 metric
    nth_neighbor = np.sort(distance_mat, axis=1)[:, 1]
    fix_group_id = (nth_neighbor < cluster_thr).astype(int).cumsum()

    # NA 값 처리
    fix_group_id = np.append(fix_group_id, fix_group_id[-1])
    ###########################################################################

    # 값 업데이트
    for rp, group_id in zip(rps, fix_group_id):
        setattr(rp, "fix_group_id", group_id)
    return rps


def discard_short_fixation(rps):
    velocity_threshold = params.velocity_threshold

    fix_groups = defaultdict(lambda: defaultdict(list))
    fix_group_id = 0

    for rp in rps[:-1]:
        if rp.speed > velocity_threshold:
            setattr(rp, "label", "saccade")
    #         print('saccade')
            rp.fix_group_id = fix_group_id
            fix_groups[fix_group_id]["x"].append(rp.x)
            fix_groups[fix_group_id]["y"].append(rp.y)
            fix_groups[fix_group_id]["timestamp"].append(rp.timestamp)
            fix_group_id += 1
        else:
            setattr(rp, "label", "fixation")
    #         print('fixation')
            rp.fix_group_id = fix_group_id
            fix_groups[fix_group_id]["x"].append(rp.x)
            fix_groups[fix_group_id]["y"].append(rp.y)
            fix_groups[fix_group_id]["timestamp"].append(rp.timestamp)
        
    for rp in rps:   
        if len(fix_groups[rp.fix_group_id]["x"]) == 1:
            del(fix_groups[rp.fix_group_id])
    
    min_fix_duration = params.min_fix_duration
        
    rf_inputs = list(map(lambda fix_group : {
        "x" : np.median(fix_group["x"]),
        "y" : np.median(fix_group["y"]),
        "timestamp" : np.max(fix_group["timestamp"]),
        "duration" : np.max(fix_group["timestamp"]) - np.min(fix_group["timestamp"])
    }, fix_groups.values()))

    rfs = [RawFixation(rf_input) for rf_input in rf_inputs if rf_input['duration'] >= min_fix_duration]

    return rfs


def get_rf(rps):
    """
    Description:

    :param rps: List<RawGazePoint> Raw Gaze Point
    :return: List<RawFixation> Raw Fixation
    """
    # Gap Fill In: 현재 모든 데이터에서 좌표가 다 있는 상태라서 굳이 구현하지 않아도 될듯
    rps = time_correction(rps)
    rps = gap_fill_in(rps)

    rps = noise_reduction(rps)
    rps = calculate_velocity(rps)
    rps = ivt_classifier(rps)
    rps = merge_adj_fixation(rps)
    rfs = discard_short_fixation(rps)
    return rfs


def run(rps):
    """
    :param rps: List<RawGazePoint> Raw gaze Point
    :return: rfs: List<RawFixation> Raw Fixation

    현재 단순히 raw fixation만 작동하게 되어 있는데, 데이터가 들어오는 것에 따라서
    전처리나 후처리 등의 과정이 추가될 수도 있는 상태.
    """

    # 전처리
    #   - Blink 처리가 아마 여기서 필요하지 않을까 생각
    # ver.0.1: 현재 데이터에서 x나 y값이 -9999로 찍히는게 존재. 이거에 대한 처리가 필요
    for rp in rps:
        if rp.x == -9999 or rp.y == -9999:
            rp.x = np.nan
            rp.y = np.nan

    # Raw Gaze Point --> Raw Fixation
    rfs = get_rf(rps)

    # 후처리(TBD)

    return rfs
