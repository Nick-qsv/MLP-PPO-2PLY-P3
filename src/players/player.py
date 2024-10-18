# backgammon/players/player.py

from enum import IntEnum


class Player(IntEnum):
    """
    An enumeration to represent the players with a simple integer value for each player.
    """

    PLAYER1 = 0
    PLAYER2 = 1
