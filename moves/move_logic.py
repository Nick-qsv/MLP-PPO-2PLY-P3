# backgammon/moves/move_logic.py

from typing import List
from ..board.board import Board
from ..players.player import Player
from .move_types import SubMove
from .conditions import valid_move, check_if_blot, is_valid_entry_at_index
from ..board.board_state import BoardState  # Add this import
from ..board.point_state import PointState  # Add this import

# Functions originally from the initial code


def get_moves_with_one_die(board: Board, die_value: int, player: Player) -> list:
    current_board_state = compute_board_state(board, player)

    if current_board_state == BoardState.NORMAL:
        return get_moves_normal(board, die_value, player)
    elif current_board_state == BoardState.ON_BAR:
        return get_moves_bar(board, die_value, player)
    elif current_board_state == BoardState.BEAR_OFF:
        return get_moves_bear_off(board, die_value, player)
    else:
        return []


def get_moves_normal(board: Board, die_value: int, player: Player) -> List[SubMove]:
    moves = []
    direction = 1 if player == Player.PLAYER1 else -1
    number_of_points = 24

    for idx in range(number_of_points):
        point = board.points[idx]
        if point != PointState.EMPTY and point[0] == PointState.OWNED:
            owned_state = point[1]
            if owned_state.player == player and owned_state.count > 0:
                destination_idx = idx + die_value * direction
                if valid_move(
                    destination_idx=destination_idx, player=player, board=board
                ):
                    sub_move = SubMove(
                        start_index=idx,
                        end_index=destination_idx,
                        hits_blot=check_if_blot(
                            index=destination_idx, player=player, board=board
                        ),
                    )
                    moves.append(sub_move)
    return moves


def get_moves_bar(board: Board, die_value: int, player: Player) -> List[SubMove]:
    moves = []
    number_of_points = 24  # Total number of points on the board
    dest_idx = (
        die_value - 1 if player == Player.PLAYER1 else number_of_points - die_value
    )

    if is_valid_entry_at_index(dest_idx, player, board):
        sub_move = SubMove(
            start_index=-1,  # Represents starting from the bar
            end_index=dest_idx,
            hits_blot=check_if_blot(index=dest_idx, player=player, board=board),
        )
        moves.append(sub_move)
    return moves


def get_moves_bear_off(board: Board, die_value: int, player: Player) -> List[SubMove]:
    move_set = []
    number_of_points = 24  # Total number of points on the board

    # Define the home board indexes for each player
    player1_indexes = list(range(18, 24))  # Player 1 exits at positions 18-23
    player2_indexes = list(range(0, 6))  # Player 2 exits at positions 0-5

    # Select exit indexes and range based on the current player
    if player == Player.PLAYER1:
        exit_indexes = player1_indexes
        exit_range = range(18, 24)  # From 18 to 23 inclusive
        direction = 1
        index_of_last_checker = 18
    else:
        exit_indexes = player2_indexes
        exit_range = range(5, -1, -1)  # From 5 down to 0 inclusive
        direction = -1
        index_of_last_checker = 5

    # Normal moves within the board
    for idx in exit_indexes:
        point = board.points[idx]
        if point != PointState.EMPTY and point[0] == PointState.OWNED:
            owned_state = point[1]
            if owned_state.player == player and owned_state.count > 0:
                destination_idx = idx + die_value * direction
                if valid_move(
                    destination_idx=destination_idx, player=player, board=board
                ):
                    sub_move = SubMove(
                        start_index=idx,
                        end_index=destination_idx,
                        hits_blot=check_if_blot(
                            index=destination_idx, player=player, board=board
                        ),
                    )
                    move_set.append(sub_move)

    # Find the index of the farthest checker from the exit point (last checker)
    for idx in exit_range:
        point = board.points[idx]
        if point != PointState.EMPTY and point[0] == PointState.OWNED:
            owned_state = point[1]
            if owned_state.player == player:
                index_of_last_checker = idx
                break

    # Handle bear-off moves
    if player == Player.PLAYER1:
        # Standard bear-off move
        if die_value >= number_of_points - index_of_last_checker:
            sub_move = SubMove(
                start_index=index_of_last_checker,
                end_index=-2,  # -2 represents bearing off
                hits_blot=False,
            )
            move_set.append(sub_move)

        # Special bear-off move
        potential_start_index = number_of_points - die_value
        if potential_start_index != index_of_last_checker:
            point = board.points[potential_start_index]
            if point != PointState.EMPTY and point[0] == PointState.OWNED:
                owned_state = point[1]
                if owned_state.player == player and owned_state.count > 0:
                    sub_move = SubMove(
                        start_index=potential_start_index, end_index=-2, hits_blot=False
                    )
                    move_set.append(sub_move)
    else:
        # Player 2 bear-off moves
        # Standard bear-off move
        if die_value >= index_of_last_checker + 1:
            sub_move = SubMove(
                start_index=index_of_last_checker, end_index=-2, hits_blot=False
            )
            move_set.append(sub_move)

        # Special bear-off move
        potential_start_index = die_value - 1
        if potential_start_index != index_of_last_checker:
            point = board.points[potential_start_index]
            if point != PointState.EMPTY and point[0] == PointState.OWNED:
                owned_state = point[1]
                if owned_state.player == player and owned_state.count > 0:
                    sub_move = SubMove(
                        start_index=potential_start_index, end_index=-2, hits_blot=False
                    )
                    move_set.append(sub_move)

    return move_set


def compute_board_state(board: Board, player: Player) -> "BoardState":
    """
    Computes the current state of the board for the given player.
    """
    if check_for_win(board, player):
        return BoardState.GAME_OVER
    if check_for_bar(board, player):
        return BoardState.ON_BAR
    if all_checkers_home(board, player):
        return BoardState.BEAR_OFF
    return BoardState.NORMAL


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
