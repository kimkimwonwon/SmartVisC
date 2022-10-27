import os
from utils.data_handler import DataHandler
import argparse
from copy import deepcopy


parser = argparse.ArgumentParser(
                    prog='SmartVC',
                    description='What the program does')

parser.add_argument("--is_sample",
                    type=str,
                    default="0",
                    )

args = parser.parse_args()


def main():
    # Mission 1: iVT Filter
    # 이건 이미 성공했다고 가정

    # Mission 2: Line Allocation
    bm_cf = deepcopy(handler.get_sample_cf())
    handler.run_alloc()
    current_cf = handler.get_sample_cf()
    print()


if __name__ == '__main__':
    path_root = os.getcwd()
    handler = DataHandler(path_root, is_sample=True)
    main()
