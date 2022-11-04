# iVT Filter parameter
## Gap fill-in
max_gap_length = 75 # 3.1.1.4 참고

## Noise Reduction
window_size = 3

## Calculate Velocity
window_len = 20 #  3.1.4 마지막 문단 참고

## absolute_threshold
velocity_threshold = 5 # 3.1.5 마지막 문단의 값과 다른 값을 현재 사용중

## Merge adjacent fixation
cluster_thr = 0.4
num_neighbor = 10
max_time_between_fixations = 75 # 3.1.6 [-1]문단 참고

## Discard short fixations
min_fix_duration = 60 # 3.1.7 참고

# Line Allocation
backward_threshold = -600