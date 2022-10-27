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

    def __str__(self):
        return f"blink{self.blink}_timestamp{self.timestamp}"


class RawFixation:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)


class CorrectedFixation:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)


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

        self.age = raw['age']
        self.gender = raw['gender']
        self.level = raw['level']

        self.wordAoiList = [WordAoi(i) for i in raw['wordAoi']]
        self.rawGazePointList = [RawGazePoint(i) for i in raw['rawGazePoint']]

        self.rawFixationList = None
        self.correctedFixationList = None

        # NOTE: sample data인 경우에는 sample 데이터에 처리된 결과를 가지도록
        if "rawFixation" in list(raw.keys()):
            self.rawFixationList = [RawFixation(i) for i in raw['rawFixation']]
        if "correctedFixation" in list(raw.keys()):
            self.correctedFixationList = [CorrectedFixation(i) for i in raw['correctedFixation']]

    def __str__(self):
        name = f"oid_{self.idx}_age_{self.age}_gender_{self.gender}"
        return name

