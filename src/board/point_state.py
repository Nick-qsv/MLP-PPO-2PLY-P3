# backgammon/board/point_state.py

from enum import Enum, auto


class PointState(Enum):
    EMPTY = auto()
    OWNED = auto()
