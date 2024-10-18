# backgammon/moves/move_types.py

from dataclasses import dataclass
from typing import List
from src.players.player import Player
from enum import IntEnum


class Position(IntEnum):
    P_0 = 0
    P_1 = 1
    P_2 = 2
    P_3 = 3
    P_4 = 4
    P_5 = 5
    P_6 = 6
    P_7 = 7
    P_8 = 8
    P_9 = 9
    P_10 = 10
    P_11 = 11
    P_12 = 12
    P_13 = 13
    P_14 = 14
    P_15 = 15
    P_16 = 16
    P_17 = 17
    P_18 = 18
    P_19 = 19
    P_20 = 20
    P_21 = 21
    P_22 = 22
    P_23 = 23
    BAR = 24
    BEAR_OFF = 25


@dataclass(frozen=True)
class SubMove:
    start: Position
    end: Position
    hits_blot: bool


@dataclass
class FullMove:
    sub_move_commands: List[SubMove]
    player: Player
