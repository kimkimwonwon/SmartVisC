import os
from utils.data_handler import DataHandler
import argparse


parser = argparse.ArgumentParser(
                    prog='SmartVC',
                    description='What the program does')

parser.add_argument("--is_sample",
                    type=str,
                    default="0",
                    )

args = parser.parse_args()


def main():
    # Phase-1 : iVT Filter & Line Allocation

    # Mission 1: iVT Filter
    bm_rf = handler.get_sample_rf()
    handler.run_ivt()
    current_rf = handler.get_sample_rf()
    print()


if __name__ == '__main__':
    path_root = os.getcwd()
    handler = DataHandler(path_root, is_sample=True)
    main()
