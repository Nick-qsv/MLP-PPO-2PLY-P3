# backgammon/moves/conditions.py

from typing import List
from src.board.board_class import Board
from src.players.player import Player
from src.board.point_state import PointState

NUMBER_OF_POINTS = 24

PLAYER_TO_INDEX = {
    Player.PLAYER1: 0,
    Player.PLAYER2: 1,
}


def valid_move(destination_idx: int, player: Player, board: Board) -> bool:
    player_idx = PLAYER_TO_INDEX[player]
    opponent_idx = 1 - player_idx

    if 0 <= destination_idx < NUMBER_OF_POINTS:
        opponent_checkers = board.points[opponent_idx, destination_idx].item()
        if opponent_checkers >= 2:
            return False
        else:
            return True
    elif destination_idx == BEAR_OFF_INDEX:
        # Bearing off
        return True
    else:
        return False


def check_if_blot(index: int, player: Player, board: Board) -> bool:
    opponent_idx = 1 - PLAYER_TO_INDEX[player]

    if 0 <= index < NUMBER_OF_POINTS:
        opponent_checkers = board.points[opponent_idx, index].item()
        if opponent_checkers == 1:
            return True
    return False


def is_valid_entry_at_index(index: int, player: Player, board: Board) -> bool:
    opponent_idx = 1 - PLAYER_TO_INDEX[player]

    if 0 <= index < NUMBER_OF_POINTS:
        opponent_checkers = board.points[opponent_idx, index].item()
        if opponent_checkers >= 2:
            return False
        else:
            return True
    else:
        return False


def check_for_bar(board: Board, player: Player) -> bool:
    player_idx = PLAYER_TO_INDEX[player]
    return board.bar[player_idx].item() > 0


def check_for_win(board: Board, player: Player) -> bool:
    player_idx = PLAYER_TO_INDEX[player]
    return board.borne_off[player_idx].item() == 15


def all_checkers_home(board: Board, player: Player) -> bool:
    home_range = range(0, 6) if player == Player.PLAYER2 else range(18, 24)
    player_idx = PLAYER_TO_INDEX[player]
    total_checkers = 0

    for idx in range(NUMBER_OF_POINTS):
        num_checkers = board.points[player_idx, idx].item()
        if num_checkers > 0:
            if idx in home_range:
                total_checkers += num_checkers
            else:
                return False  # Checker outside home board

    # Check for any checkers on the bar
    if board.bar[player_idx].item() > 0:
        return False

    # Include borne off checkers
    borne_off_checkers = board.borne_off[player_idx].item()

    # Check if total checkers (home + borne off) == 15
    return total_checkers + borne_off_checkers == 15
