import os
import json
from pprint import pprint
import numpy as np

from utils.data import Visc
from utils import iVT
from utils import allocation
import glob
from tqdm import tqdm


class DataHandler:
    def __init__(self, pr, is_sample=True, sample_id=0, dat_fn="343FullData.json"):
        self.path_root = pr
        self.path_data = f"{pr}/data/raw"
        self.meta = dict()

        # Used only for raw data
        self.num_fixation = 0
        self.num_saccade = 0

        self.sample_id = sample_id

        num_excluded = 0
        with open(f"{self.path_data}/{dat_fn}", encoding="utf-8") as f:
            dat_list = json.load(f)
        f.close()
        if type(dat_list) == list:
            for i, tmp in enumerate(dat_list):
                if len(tmp["rawGazePoint"]) == 0:
                    dat_list.pop(i)
                    num_excluded += 1
        else:
            dat_list = [dat_list]
        if not is_sample:
            self.meta["status"] = "load"
            self.meta["dsrc"] = "raw"
            print("Every data has been loaded!")
            # 문제가 되는 데이터
            dat_list.pop(28)
        else:
            if self.sample_id > len(dat_list)-1:
                self.sample_id = len(dat_list) - 1
            dat_list = [dat_list[self.sample_id]]
            self.sample_id = 0
            self.meta["status"] = "DONE"
            self.meta["dsrc"] = "sample"
            print(f"{self.sample_id}th data is loaded")

        print(f"Excluded data : {num_excluded}")
        self.data = [Visc(i) for i in dat_list]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f"Status{self.meta['status']}_DataSource{self.meta['dsrc']}"

    # STEP1
    def run_ivt(self):
        #
        prog = tqdm(self.data)
        for dat in prog:
            raw_fixation = iVT.run(dat.rawGazePointList)
            setattr(dat, "rawFixationList", raw_fixation)
            prog.set_description("iVT Filter Progress")

        self.meta['status'] = "iVT"

    # STEP2
    def run_alloc(self):
        prog = tqdm(self.data)
        for dat in prog:
            corrected_fixation = allocation.run(dat.rawFixationList, dat.wordAoiList)
            setattr(dat, "correctedFixationList", corrected_fixation)
            prog.set_description("Line Allocation Progress")

        self.meta['status'] = "DONE"

    # RUN
    def run(self):
        self.run_ivt()

        self.run_alloc()

    def get_sample_all(self):
        return self.data[self.sample_id]

    def get_sample_rp(self):
        return self.data[self.sample_id].rawGazePointList

    def get_sample_rf(self):
        return [i for i in self.data[self.sample_id].rawFixationList if not i.ignore]

    def get_sample_cf(self):
        return self.data[self.sample_id].correctedFixationList

    def get_resolution(self):
        return self.data[self.sample_id].screenResolution

    def get_word_aoi(self):
        return self.data[self.sample_id].wordAoiList

    def get_gaze_point_dist(self):
        bpoints = self.data[self.sample_id].boundaryPoints
        gazes = np.array([list(bpoint.get_gaze_coors()) for bpoint in bpoints])
        targets = np.array([list(bpoint.get_target_coors()) for bpoint in bpoints])
        bias = targets - gazes
        bias = bias[:, 1]
        return bias


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
