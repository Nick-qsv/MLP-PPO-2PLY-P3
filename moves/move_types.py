# backgammon/moves/move_types.py

from dataclasses import dataclass
from typing import List
from ..players.player import Player


@dataclass(frozen=True)
class SubMove:
    start_index: int
    end_index: int
    hits_blot: bool


@dataclass
class FullMove:
    sub_move_commands: List[SubMove]
    player: Player
