import os
from utils.data_handler import DataHandler
import argparse


parser = argparse.ArgumentParser(
                    prog='SmartVC',
                    description='What the program does')

parser.add_argument("--is_sample",
                    type=str,
                    default="1",
                    )

args = parser.parse_args()


def main():
    # Phase-1 : iVT Filter & Line Allocation

    # Mission 1: iVT Filter
    handler.run_ivt()

    # Mission 2: Line Allocation
    handler.run_alloc()

    # Phase-2 : Metric(TBD)

    # Mission 3: Metric


if __name__ == '__main__':
    path_root = os.getcwd()
    handler = DataHandler(path_root, is_sample=args.is_sample == "1")
    main()
