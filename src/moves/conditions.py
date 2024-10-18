# backgammon/moves/conditions.py

from typing import List
from src.board.board_class import Board
from src.players.player import Player
from src.constants import PLAYER_TO_INDEX, BEAR_OFF_INDEX, NUMBER_OF_POINTS
from src.board.immutable_board import ImmutableBoard
from src.moves.move_types import Position


def valid_move(destination_idx: int, player: Player, board: ImmutableBoard) -> bool:
    """
    Checks if moving to the destination index is a valid move for the player on the given ImmutableBoard.

    Args:
        destination_idx (int): The index of the destination point.
        player (Player): The player making the move.
        board (ImmutableBoard): The current state of the board.

    Returns:
        bool: True if the move is valid, False otherwise.
    """
    player_idx = player.value  # Get player's index (0 or 1)
    opponent_idx = 1 - player_idx  # Opponent's index

    if 0 <= destination_idx < NUMBER_OF_POINTS:
        # Access opponent's checkers at the destination point
        opponent_checkers = board.tensor[opponent_idx, destination_idx].item()
        if opponent_checkers >= 2:
            return False  # Destination is blocked by opponent
        else:
            return True
    elif destination_idx == Position.BEAR_OFF.value:
        # Bearing off is always allowed if within rules
        return True
    else:
        return False  # Invalid destination index


def check_if_blot(index: int, player: Player, board: ImmutableBoard) -> bool:
    """
    Checks if there is a blot (exactly one opponent checker) at the specified index.

    Args:
        index (int): The point index to check for a blot.
        player (Player): The current player.
        board (ImmutableBoard): The current state of the board.

    Returns:
        bool: True if there is a blot at the index, False otherwise.
    """
    opponent_idx = 1 - player.value  # Opponent's index

    if 0 <= index < NUMBER_OF_POINTS:
        opponent_checkers = board.tensor[opponent_idx, index].item()
        if opponent_checkers == 1:
            return True  # There is a blot
    return False


def is_valid_entry_at_index(index: int, player: Player, board: ImmutableBoard) -> bool:
    """
    Determines if the player can enter a checker at the specified index.

    Args:
        index (int): The point index where the checker is to be entered.
        player (Player): The player attempting to enter the checker.
        board (ImmutableBoard): The current state of the board.

    Returns:
        bool: True if the entry is valid, False otherwise.
    """
    opponent_idx = 1 - player.value  # Opponent's index

    if 0 <= index < 24:  # Valid point index
        opponent_checkers = board.tensor[opponent_idx, index].item()
        if opponent_checkers >= 2:
            return False  # Entry point is blocked
        else:
            return True
    else:
        return False  # Invalid index for entry


def check_for_bar(board: ImmutableBoard, player: Player) -> bool:
    """
    Checks if the player has any checkers on the bar.

    Args:
        board (ImmutableBoard): The current state of the board.
        player (Player): The player to check for checkers on the bar.

    Returns:
        bool: True if the player has checkers on the bar, False otherwise.
    """
    # Bar is on channel 2, index corresponds to player.value
    return board.tensor[2, player.value].item() > 0


def check_for_win(board: ImmutableBoard, player: Player) -> bool:
    """
    Checks if the player has borne off all checkers, indicating a win.

    Args:
        board (ImmutableBoard): The current state of the board.
        player (Player): The player to check for a win.

    Returns:
        bool: True if the player has won, False otherwise.
    """
    # Borne-off checkers are on channel 3, index corresponds to player.value
    return board.tensor[3, player.value].item() == 15


def all_checkers_home(board: ImmutableBoard, player: Player) -> bool:
    """
    Determines if all of the player's checkers are in the home board or have been borne off.

    Args:
        board (ImmutableBoard): The current state of the board.
        player (Player): The player to check.

    Returns:
        bool: True if all checkers are home or borne off, False otherwise.
    """
    # Define home range based on player
    if player == Player.PLAYER2:
        home_range = range(0, 6)  # Points 0-5 for PLAYER2
    else:
        home_range = range(18, 24)  # Points 18-23 for PLAYER1

    player_idx = player.value  # Player's index
    total_checkers = 0

    for idx in range(24):  # Iterate over all points
        num_checkers = board.tensor[player_idx, idx].item()
        if num_checkers > 0:
            if idx in home_range:
                total_checkers += num_checkers
            else:
                return False  # Checker is outside the home board

    # Check for any checkers on the bar (channel 2)
    if board.tensor[2, player_idx].item() > 0:
        return False  # Checkers are on the bar

    # Include borne-off checkers (channel 3)
    borne_off_checkers = board.tensor[3, player_idx].item()

    # Total checkers should be 15 (home + borne off)
    return total_checkers + borne_off_checkers == 15
