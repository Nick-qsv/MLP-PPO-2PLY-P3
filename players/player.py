# backgammon/players/player.py

from enum import Enum, auto


class Player(Enum):
    """
    An enumeration to represent the players.
    """

    PLAYER1 = auto()
    PLAYER2 = auto()
