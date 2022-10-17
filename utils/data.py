class WordBox:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)
        # self.x = raw['x']
        # self.y = raw['y']
        # self.width = raw['width']
        # self.height = raw['height']


class WordAoi:
    def __init__(self, raw: dict):
        self.idx_visc = raw['_id']["$oid"]
        self.line = raw['line']
        self.word = raw['word']
        self.order = raw['order']
        self.wordBox = WordBox(raw['wordBox'])


class RawGazePoint:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)
        # self.idx_visc = raw['_id']["$oid"]
        # self.blink = raw['blink']
        # self.faceDistance = raw['faceDistance']
        # self.timestamp = raw['timestamp']
        # self.x = raw['x']
        # self.y = raw['y']


class RawFixation:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)
        # self.idx_visc = raw['_id']["$oid"]
        # self.x = raw['x']
        # self.y = raw['y']
        # self.timestamp = raw['timestamp']
        # self.duration = raw['duration']


class CorrectedFixation:
    def __init__(self, raw: dict):
        for k, v in raw.items():
            setattr(self, k, v)
        # self.idx_visc = raw['_id']["$oid"]
        # self.x = raw['x']
        # self.y = raw['y']
        # self.timestamp = raw['timestamp']
        # self.duration = raw['duration']
        # self.line = raw['line']
        # self.order = raw['order']


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

        # Processed Data Structure
        self.rawFixationList = [RawFixation(i) for i in raw['rawFixation']]
        self.correctedFixationList = [CorrectedFixation(i) for i in raw['correctedFixation']]

    def __str__(self):
        name = f"oid_{self.idx}_age_{self.age}_gender_{self.gender}"
        return name
