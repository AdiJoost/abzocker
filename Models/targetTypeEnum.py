from enum import Enum

class TargetType(Enum):
    TIMESTAMP = 0,
    ADJ_CLOSE = 1,
    CLOSE = 2,
    HIGH = 3,
    LOW = 4,
    OPEN = 5,
    VOLUME = 6,