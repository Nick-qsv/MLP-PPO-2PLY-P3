from typing import List
from src.board.immutable_board import ImmutableBoard
from src.moves.move_types import FullMove
from src.players import Player
from src.moves.handle_moves import handle_non_doubles, handle_doubles


def get_all_possible_moves(
    player: Player, board: ImmutableBoard, roll_result: List[int]
) -> List[FullMove]:
    """
    Generates all possible full moves for the given player based on the roll result.

    Parameters:
    - player (Player): The player for whom to generate moves.
    - board_copy (ImmutableBoard): The current state of the board.
    - roll_result (List[int]): The result of the dice roll.

    Returns:
    - List[FullMove]: A list of all possible full move sequences.
    """
    full_moves = []
    unique_boards = set()
    roll = roll_result[:]

    if roll[0] != roll[1]:
        # Sort the dice in descending order unless reversed
        roll_sorted = sorted(roll, reverse=True)

        # Handle non-double dice rolls in sorted order
        handle_non_doubles(
            board=board,
            roll=roll_sorted,
            full_moves=full_moves,
            unique_boards=unique_boards,
            player=player,
        )

        # If no moves were found or more than one move sequence exists,
        # handle the reverse order of dice
        if not full_moves or not (
            len(full_moves) == 1 and len(full_moves[0].sub_move_commands) == 1
        ):
            handle_non_doubles(
                board=board,
                roll=roll_sorted,
                full_moves=full_moves,
                unique_boards=unique_boards,
                player=player,
                reverse=True,  # Reverse the dice order
            )
    else:
        # Handle double dice rolls
        handle_doubles(
            board=board,
            die_value=roll[0],  # Both dice have the same value
            full_moves=full_moves,
            unique_boards=unique_boards,
            player=player,
        )

    return full_moves
