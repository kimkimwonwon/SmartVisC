import os
import json
from pprint import pprint

from utils.data import Visc
from utils import iVT
from utils import allocation
import glob


class DataHandler:
    def __init__(self, pr, is_sample=True):
        self.path_root = pr
        self.path_data = f"{pr}/data/raw"
        self.meta = dict()

        if is_sample:
            with open(f"{pr}/data/sample.json", encoding="utf-8") as f:
                dat_list = json.load(f)
            f.close()
            self.meta["status"] = "DONE"
            self.meta["dsrc"] = "sample"
            print("Sample data is loaded")
        else:
            # TODO: Add data
            # Assumption: RawGazePoint, TextMetaData, WordAOI were given
            # Data Structure of dat:
            #   - RawGazePoint
            #   - TextMetaData
            #   - WordAOI
            dat_list = []
            flist = glob.glob(f"{self.path_data}/*.json")
            for fn in flist:
                # NOTE: 여기에선 사람마다 파일이 나눠져있다고 가정한 것이고, 한 파일에 다 저장된 형태라면
                # 바로 datlist로 할당해버리면 됨
                with open(f"{fn}") as f:
                    dat = json.load(f)
                f.close()

                if type(dat) == list:
                    dat_list.extend(dat)
                else:
                    dat_list.append(dat)
            print("Data does not exist!")
            self.meta["status"] = "load"
            self.meta["dsrc"] = "raw"

        self.data = [Visc(i) for i in dat_list]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f"Status{self.meta['status']}_DataSource{self.meta['dsrc']}"

    # STEP1
    def run_ivt(self):
        #
        for dat in self.data:
            raw_fixation = iVT.run(dat.rawGazePointList)
            setattr(dat, "rawFixationList", raw_fixation)

        self.meta['status'] = "iVT"

    # STEP2
    def run_alloc(self):
        for dat in self.data:
            corrected_fixation = allocation.run(dat.rawFixationList, dat.wordAoiList)
            setattr(dat, "correctedFixationList", corrected_fixation)

        self.meta['status'] = "DONE"

    # RUN
    def run(self):
        self.run_ivt()

        self.run_alloc()

    def get_sample_all(self):
        return self.data[0]

    def get_sample_rp(self):
        return self.data[0].rawGazePointList

    def get_sample_rf(self):
        return self.data[0].rawFixationList

    def get_sample_cf(self):
        return self.data[0].correctedFixationList

    def get_resolution(self):
        return self.data[0].screenResolution

    def get_word_aoi(self):
        return self.data[0].wordAoiList


if __name__ == '__main__':
    os.chdir('..')
    path_root = os.getcwd()
    handler = DataHandler(path_root)
    print("The Number of data : ", len(handler))

    sample = handler.get_sample_all()
    print("ID : ", sample)
    pprint(sample.__dict__.keys())

    print("WordAOI : ", sample.wordAoiList[0])

    handler.run_alloc()
    print()
