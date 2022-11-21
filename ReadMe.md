# version log

## ver.1.1

1. data.py
    - 새로 업데이트된 데이터에 대해 추가된 것들은 다음과 같습니다.
        - 현재 raw 데이터에 있지만 iVT, LineAllo에 무관했던 데이터들 중에서, 추후 metric에 필요해 보이는 데이터를 활성화했습니다
        - wordAoi는 우선 데이터가 다 존재한다고 가정해서 업데이트했습니다.
        - BoundaryPoint는 새로 추가된 데이터로, calibration 에 관련된 정보를 담고 있습니다.
2. data_handler.py
    - 새로 업데이트 된 데이터에 맞춰 로드하는 루틴을 바꿨습니다.
        - 현재 있는 데이터로 로드가 전부 진행됩니다.
        - 컴퓨터에서 오래 걸리는 경우, 일부 sample만 뽑아서 보실 경우, 다음과 같이 진행하시면 됩니다.
            - hander를 initialization을 하는 경우 is_sample=True로 추가하시고, sample_id=원하시는 숫자. 로 하시면 됩니다.
            - 간혹 sample_id가 전체 개수보다 넘은 경우 가장 마지막 sample을 사용하도록 합니다.
3. visual.py
    - 기존 돌아가는 방식과 동일하게 진행됩니다.
    - 다만 순서대로 어떻게 점이 찍히는지 확인하실 분들은 다음과 같이 진행하시면 됩니다.
        - 찍고 싶으신 점들로 plot_points()를 실행하실 줄에서 is_save=True, fig=fig를 추가하시면 됩니다.

# Title

# Branch

- iVT
- iVT-check_result
- line

# Directory & Module

## Data

Data is not uploaded in github repository.

## Utils

- correction.py: **line allocation mission**
- data.py: data structure
- data_handler.py: Handler class
- iVT.py: **iVT Filter customization mission**
- metric.py: **custom metric mission**


# Sample Status

## iVT Filter
![사진](/figure/iVT_status.png)

## Line Allocation
![사진](/figure/lineAllo_status.png)