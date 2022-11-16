# iVT Filter parameter
## Gap fill-in
max_gap_length = 75 # 3.1.1.4 참고
max_fill_length = 250
max_y_distance = 100
max_x_distance = 100

## Noise Reduction
window_size = 3

## Calculate Velocity
window_len = 2 #  3.1.4 마지막 문단 참고

## absolute_threshold
velocity_threshold = 1.5 # 3.1.5 마지막 문단의 값과 다른 값을 현재 사용중

# ## Merge adjacent fixation (don't use in Ver.1)
# cluster_thr = 7
# num_neighbor = 10
# max_time_between_fixations = 75 # 3.1.6 [-1]문단 참고

## Discard short fixations
min_fix_duration = 60 # 3.1.7 참고

# Line Allocation
backward_threshold = -300