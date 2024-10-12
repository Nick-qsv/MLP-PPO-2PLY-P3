# backgammon/moves/conditions.py

from typing import List
from src.board.board import Board
from src.players.player import Player
from src.board.point_state import PointState

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


def check_for_bar(board: Board, player: Player) -> bool:
    """
    Checks if the player has any checkers on the bar.
    """
    return board.bar[player] > 0


def check_for_win(board: Board, player: Player) -> bool:
    """
    Checks if the player has won the game.
    """
    return board.borne_off[player] == 15


def all_checkers_home(board: Board, player: Player) -> bool:
    """
    Checks if all of the player's checkers are in the home board.
    """
    # Define the home range for each player
    home_range = range(0, 6) if player == Player.PLAYER2 else range(18, 24)
    home_checker_count = 0

    # Iterate through the points on the board
    for index, point in enumerate(board.points):
        if isinstance(point, tuple) and point[0] == PointState.OWNED:
            owned_state = point[1]
            if owned_state.player == player:
                if index in home_range:
                    # Add the number of checkers at this point to the homeCheckerCount
                    home_checker_count += owned_state.count
                else:
                    # If any checker of the current player is found outside the home range, return False
                    return False

    # Get the count of checkers that are borne off for the current player
    borne_off_count = board.borne_off.get(player, 0)

    # Check if the sum of checkers in the home range plus the number of checkers borne off equals 15
    return home_checker_count + borne_off_count == 15
