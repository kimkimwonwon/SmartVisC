import os
from utils.data_handler import DataHandler


def main():
    print(len(handler))
    print(handler.get_sample())
    print()


if __name__ == '__main__':
    path_root = os.getcwd()
    handler = DataHandler(path_root)
    main()
