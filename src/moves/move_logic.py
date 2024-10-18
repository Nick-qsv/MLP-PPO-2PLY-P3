# backgammon/moves/move_logic.py

from typing import List
from src.board.board_class import Board
from src.players.player import Player
from src.moves.move_types import SubMove
from .conditions import (
    valid_move,
    check_if_blot,
    is_valid_entry_at_index,
    check_for_win,
    check_for_bar,
    all_checkers_home,
)
from src.board.board_state import BoardState
from src.constants import BAR_INDEX, PLAYER_TO_INDEX, BEAR_OFF_INDEX
from src.board.immutable_board import ImmutableBoard
from src.constants import NUMBER_OF_POINTS
from src.moves.move_types import Position


def get_moves_with_one_die(
    board: ImmutableBoard, die_value: int, player: Player
) -> list:
    """
    Generates all possible sub-moves for a player given a single die value based on the current board state.

    Args:
        board (ImmutableBoard): The current state of the board.
        die_value (int): The value of the die rolled.
        player (Player): The player for whom to generate moves.

    Returns:
        list: A list of possible SubMove instances.
    """
    # Determine the current state of the board for the player
    current_board_state = compute_board_state(board, player)

    if current_board_state == BoardState.NORMAL:
        return get_moves_normal(board, die_value, player)
    elif current_board_state == BoardState.ON_BAR:
        return get_moves_bar(board, die_value, player)
    elif current_board_state == BoardState.BEAR_OFF:
        return get_moves_bear_off(board, die_value, player)
    else:
        return []  # No moves available (e.g., GAME_OVER)


def get_moves_normal(
    board: ImmutableBoard, die_value: int, player: Player
) -> List[SubMove]:
    """
    Retrieves all valid normal sub-moves for the player based on the die value.

    Args:
        board (ImmutableBoard): The current state of the board.
        die_value (int): The value of the die rolled.
        player (Player): The player for whom to generate moves.

    Returns:
        List[SubMove]: A list of valid SubMove instances.
    """
    moves = []
    player_idx = player.value  # 0 for PLAYER1, 1 for PLAYER2
    direction = (
        1 if player == Player.PLAYER1 else -1
    )  # Movement direction based on player

    for idx in range(NUMBER_OF_POINTS):
        own_checkers = board.tensor[player_idx, idx].item()
        if own_checkers > 0:
            # Calculate destination index based on direction and die value
            destination_idx = idx + die_value * direction

            # Only consider moves within the board (exclude bear-off)
            if 0 <= destination_idx < NUMBER_OF_POINTS:
                if valid_move(
                    destination_idx=destination_idx, player=player, board=board
                ):
                    # Determine if the destination has a blot to hit
                    hits_blot = check_if_blot(
                        index=destination_idx, player=player, board=board
                    )

                    # Create SubMove with Position enums
                    start_position = Position(idx)
                    end_position = Position(destination_idx)

                    sub_move = SubMove(
                        start=start_position, end=end_position, hits_blot=hits_blot
                    )
                    moves.append(sub_move)

    return moves


def get_moves_bar(
    board: ImmutableBoard, die_value: int, player: Player
) -> List[SubMove]:
    """
    Generates all valid sub-moves for a player who has checkers on the bar based on the die value.

    Args:
        board (ImmutableBoard): The current state of the board.
        die_value (int): The value of the die rolled.
        player (Player): The player for whom to generate moves.

    Returns:
        List[SubMove]: A list of valid SubMove instances for entering from the bar.
    """
    moves = []
    # Determine the destination index based on player perspective
    if player == Player.PLAYER1:
        dest_idx = die_value - 1  # PLAYER1 enters from points 0 to 5
    else:
        dest_idx = 24 - die_value  # PLAYER2 enters from points 23 to 18

    # Ensure dest_idx is within the player's home board
    if player == Player.PLAYER1:
        valid_home_indices = range(0, 6)  # 0 to 5 inclusive
    else:
        valid_home_indices = range(18, 24)  # 18 to 23 inclusive

    # Check if the destination index is within the home board
    if dest_idx in valid_home_indices:
        # Check if the entry point is valid (not blocked by two or more opponent checkers)
        if is_valid_entry_at_index(index=dest_idx, player=player, board=board):
            start_position = Position.BAR  # Starting from the bar
            end_position = Position(dest_idx)  # Destination within home board

            # Determine if the destination has a blot to hit
            hits_blot = check_if_blot(index=dest_idx, player=player, board=board)

            sub_move = SubMove(
                start=start_position, end=end_position, hits_blot=hits_blot
            )
            moves.append(sub_move)

    return moves


def get_moves_bear_off(board: Board, die_value: int, player: Player) -> List[SubMove]:
    move_set = []
    player_idx = PLAYER_TO_INDEX[player]

    if player == Player.PLAYER1:
        home_indexes = list(range(18, 24))
        direction = 1
        last_checker_idx = 18
    else:
        home_indexes = list(range(0, 6))
        direction = -1
        last_checker_idx = 5

    # Normal moves within the home board
    for idx in home_indexes:
        own_checkers = board.points[player_idx, idx].item()
        if own_checkers > 0:
            destination_idx = idx + die_value * direction
            if valid_move(destination_idx=destination_idx, player=player, board=board):
                sub_move = SubMove(
                    start_index=idx,
                    end_index=destination_idx,
                    hits_blot=check_if_blot(
                        index=destination_idx, player=player, board=board
                    ),
                )
                move_set.append(sub_move)

    # Find the index of the farthest checker from the exit point (last checker)
    if player == Player.PLAYER1:
        exit_range = range(18, 24)
        for idx in exit_range:
            if board.points[player_idx, idx].item() > 0:
                last_checker_idx = idx
                break
    else:
        exit_range = range(5, -1, -1)
        for idx in exit_range:
            if board.points[player_idx, idx].item() > 0:
                last_checker_idx = idx
                break

    # Handle bear-off moves
    if player == Player.PLAYER1:
        if last_checker_idx + die_value * direction >= 24:
            sub_move = SubMove(
                start_index=last_checker_idx, end_index=BEAR_OFF_INDEX, hits_blot=False
            )
            move_set.append(sub_move)

        potential_start_index = 24 - die_value
        if potential_start_index != last_checker_idx:
            if board.points[player_idx, potential_start_index].item() > 0:
                sub_move = SubMove(
                    start_index=potential_start_index,
                    end_index=BEAR_OFF_INDEX,
                    hits_blot=False,
                )
                move_set.append(sub_move)
    else:
        if last_checker_idx + die_value * direction < 0:
            sub_move = SubMove(
                start_index=last_checker_idx, end_index=BEAR_OFF_INDEX, hits_blot=False
            )
            move_set.append(sub_move)

        potential_start_index = die_value - 1
        if potential_start_index != last_checker_idx:
            if board.points[player_idx, potential_start_index].item() > 0:
                sub_move = SubMove(
                    start_index=potential_start_index,
                    end_index=BEAR_OFF_INDEX,
                    hits_blot=False,
                )
                move_set.append(sub_move)

    return move_set


def compute_board_state(board: ImmutableBoard, player: Player) -> "BoardState":
    """
    Determines the current state of the board for the specified player.

    Args:
        board (ImmutableBoard): The current state of the board.
        player (Player): The player for whom to compute the board state.

    Returns:
        BoardState: The current state of the board (NORMAL, ON_BAR, BEAR_OFF, GAME_OVER).
    """
    if check_for_win(board, player):
        return BoardState.GAME_OVER  # Player has won the game
    if check_for_bar(board, player):
        return BoardState.ON_BAR  # Player has checkers on the bar
    if all_checkers_home(board, player):
        return BoardState.BEAR_OFF  # Player can start bearing off
    return BoardState.NORMAL  # Player is in a normal state
