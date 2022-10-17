import os
import json
from pprint import pprint

from utils.data import Visc
from utils import iVT
from utils import correction
import glob


class DataHandler:
    def __init__(self, pr, is_sample=True):
        self.path_root = pr
        self.path_data = f"{pr}/data"

        if is_sample:
            with open(f"{self.path_data}/sample.json", encoding="utf-8") as f:
                datList = json.load(f)
            f.close()
        else:
            # TODO: Add data
            # Assumption: RawGazePoint, TextMetaData, WordAOI were given
            # Data Structure of dat:
            #   - RawGazePoint
            #   - TextMetaData
            #   - WordAOI
            datList = []

            flist = glob.glob(self.path_data)
            for fn in flist:
                # NOTE: Temporary Backup
                with open(f"{fn}") as f:
                    dat = json.load(f)
                wordAoi = dat['WordAoi']
                dat = iVT.run(dat['RawGazePoint'], wordAoi)

                true_fixation = dat['TrueFixation']
                dat = correction.run(true_fixation, wordAoi)
                datList.append(dat)
            print("Data does not exist!")
        self.data = [Visc(i) for i in datList]

    def __len__(self):
        return len(self.data)

    def get_sample(self):
        return self.data[0]


if __name__ == '__main__':
    os.chdir('..')
    path_root = os.getcwd()
    handler = DataHandler(path_root)
    print(len(handler))
    print(handler.get_sample())
    print()