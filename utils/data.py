import utils.const as const


class WordBox:
    def __init__(self, raw: dict):
        self.height = raw["height"]
        self.width = raw["width"]
        # ver.0.2 : 단어 중심이 오도록 정렬
        # ver.1.1 : 오히려 단어 정렬하는 과정에서 단어들이 잘못 정렬되어서 버려야함.
        self.x = raw["x"]
        self.y = raw["y"]

    def __str__(self):
        return f"X{self.x}_Y{self.y}_H{self.height}_W{self.width}"


class WordAoi:
    def __init__(self, raw: dict):
        self.idx = raw['_id']["$oid"]
        self.line = raw['line'] - 1
        self.word = raw['word']
        self.order = raw['order'] - 1
        self.wordBox = WordBox(raw['wordBox'])
        # ver.1.1 : 해당 단어의 개수도 파악
        self.word_cnt = len(self.word)

    def __str__(self):
        return f"{self.word}_line{self.line}_order{self.order}"


class RawGazePoint:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            if k == "timestamp":
                v = int(v)
            setattr(self, k, v)
        self.speed = None
        self.label = None
        self.fix_group_id = None

    def __str__(self):
        return f"blink{self.blink}_timestamp{self.timestamp}"


class RawFixation:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)
        self.is_backward = None
        self.ftype = None
        self.segment_id = None
        self.line_id = None
        self.order_id = None
        self.ignore = False

    def __str__(self):
        return f"Timestamp{self.timestamp}_X{self.x}_Y{self.y}"


class CorrectedFixation:
    def __init__(self, raw: dict):
        self.timestamp = raw['timestamp']
        self.line = raw['line']
        self.order = raw['order']
        self.duration = raw['duration']
        self.x = raw['x']
        self.y = raw['y']
        self.ftype = None

    def __str__(self):
        return f"Timestamp{self.timestamp}_line{self.line}_order{self.order}"


class TextMetaData:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)


# ver.1.1 : 새로 추가된 데이터용 class입니다.
class BoundaryPoint:
    def __init__(self, raw: dict):
        self.target = raw['target']
        self.gaze = raw['gaze']
        self.idx = raw["_id"]["$oid"]


class Visc:
    """
    Current structure has potential OOM problem
    """
    def __init__(self, raw: dict):
        self.idx = raw['_id']["$oid"]
        self.screenResolution = raw['deviceInformation']['screenResolution']

        # ver.1.1 : 현재 raw 데이터에 있지만 iVT, LineAllo에 무관했던 데이터들 중에서, 추후 metric에 필요해 보이는 데이터를 활성화했습니다
        self.age = raw['age']
        self.gender = raw['gender']
        self.level = raw['level']
        self.title = raw['title']
        self.device_info = raw["deviceInformation"]
        self.text_meta = TextMetaData(raw["textMetadata"])

        # ver.1.1: 새로 추가된 데이터로, calibration 에 관련된 정보를 담고 있습니다.
        self.boundaryPoints = [BoundaryPoint(i) for i in raw["boundaryPoint"]]
        # ver.1.1: wordAoi는 우선 데이터가 다 존재한다고 가정해서 업데이트했습니다.
        self.wordAoiList = [WordAoi(i) for i in raw['wordAoi']]

        self.rawGazePointList = [RawGazePoint(i) for i in raw['rawGazePoint']]
        self.rawFixationList = None
        self.correctedFixationList = [CorrectedFixation(i) for i in raw['correctedFixation']]

        # ver.1.1 : 스크린 크기를 기록하는 부분입니다.
        const.screen_height = self.device_info['screenResolution']['height']
        const.screen_width = self.device_info['screenResolution']['width']
        const.font_size = self.text_meta["fontSize"]

    def __str__(self):
        name = f"oid_{self.idx}"
        return name
