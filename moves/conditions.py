# backgammon/moves/conditions.py

from typing import List
from ..board.board import Board
from ..players.player import Player
from ..board.point_state import PointState

NUMBER_OF_POINTS = 24


def valid_move(destination_idx: int, player: Player, board: Board) -> bool:
    if 0 <= destination_idx < NUMBER_OF_POINTS:
        dest_state = board.points[destination_idx]
        if dest_state == PointState.EMPTY:
            return True
        elif dest_state[0] == PointState.OWNED:
            owned_state = dest_state[1]
            if owned_state.player == player:
                return True
            elif owned_state.count == 1:
                return True  # You can hit a blot
            else:
                return False
    else:
        return False


def check_if_blot(index: int, player: Player, board: Board) -> bool:
    if 0 <= index < NUMBER_OF_POINTS:
        point = board.points[index]
        if point == PointState.EMPTY:
            return False
        elif point[0] == PointState.OWNED:
            owned_state = point[1]
            if owned_state.player == player:
                return False
            elif owned_state.count == 1:
                return True  # It's a blot
            else:
                return False
    else:
        return False


def is_valid_entry_at_index(index: int, player: Player, board: Board) -> bool:
    if 0 <= index < NUMBER_OF_POINTS:
        point = board.points[index]
        if point == PointState.EMPTY:
            return True
        elif point[0] == PointState.OWNED:
            owned_state = point[1]
            if owned_state.player == player:
                return True
            elif owned_state.count == 1:
                return True  # You can hit a blot
            else:
                return False
    else:
        return False
