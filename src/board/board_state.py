# backgammon/board/board_state.py

from enum import Enum, auto


class BoardState(Enum):
    NORMAL = auto()
    ON_BAR = auto()
    BEAR_OFF = auto()
    GAME_OVER = auto()
