from enum import Enum


class AugOp(Enum):
    DROPOUT = 0
    INSERT = 1
    CORRUPT = 2
    IDENTITY = 3
