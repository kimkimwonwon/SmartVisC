class WordBox:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)

    def __str__(self):
        return f"X{self.x}_Y{self.y}_H{self.height}_W{self.width}"


class WordAoi:
    def __init__(self, raw: dict):
        self.idx_visc = raw['_id']["$oid"]
        self.line = raw['line']
        self.word = raw['word']
        self.order = raw['order']
        self.wordBox = WordBox(raw['wordBox'])

    def __str__(self):
        return f"{self.word}_line{self.line}_order{self.order}"


class RawGazePoint:
    def __init__(self, raw: dict):
        for k, v in raw.items():
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
        self.line_group_id = None

    def __str__(self):
        return f"Timestamp{self.timestamp}_X{self.x}_Y{self.y}"


class CorrectedFixation:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)

    def __str__(self):
        return f"Timestamp{self.timestamp}_line{self.line}_order{self.order}"


class TextMetaData:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)


class Visc:
    """
    Current structure has potential OOM problem

    """
    def __init__(self, raw: dict):
        self.idx = raw['_id']["$oid"]
        self.screenResolution = raw['deviceInformation']['screenResolution']

        # NOTE: sample.json에는 있지만 ivt-sample-data.json에는 없는 것들
        # 다만 level은 향후 metric에 필요할 것으로 판단
        # self.age = raw['age']
        # self.gender = raw['gender']
        # self.level = raw['level']
        self.level = None
        
        self.rawGazePointList = [RawGazePoint(i) for i in raw['rawGazePoint']]
        # TODO: 원래 있는 것으로 되어있었는데 현재 ivt-sample-data.json 에 없는 상태!! 수정 필요
        if "wordAoi" in list(raw.keys()):
            self.wordAoiList = [WordAoi(i) for i in raw['wordAoi']]
        else:
            self.wordAoiList = None

        self.rawFixationList = None
        self.correctedFixationList = None

        # NOTE: sample data인 경우에는 sample 데이터에 처리된 결과를 가지도록
        if "rawFixation" in list(raw.keys()):
            self.rawFixationList = [RawFixation(i) for i in raw['rawFixation']]
        if "correctedFixation" in list(raw.keys()):
            self.correctedFixationList = [CorrectedFixation(i) for i in raw['correctedFixation']]

    def __str__(self):
        name = f"oid_{self.idx}"
        return name

