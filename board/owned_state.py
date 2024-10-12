# backgammon/board/owned_state.py

from dataclasses import dataclass
from ..players.player import Player


@dataclass(frozen=True)
class OwnedState:
    """
    A dataclass to represent the ownership state of a point,
    including the count of owned points and the player who owns them.
    """

    count: int
    player: Player
