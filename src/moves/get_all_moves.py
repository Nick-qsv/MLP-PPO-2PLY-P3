from typing import List
from src.board.immutable_board import ImmutableBoard
from src.moves.move_types import FullMove
from src.players import Player
from src.moves.handle_moves import handle_non_doubles, handle_doubles
from src.utils.decorators import profile, profiling_data


@profile
def get_all_possible_moves(
    player: Player, board: ImmutableBoard, roll_result: List[int]
) -> List[FullMove]:
    """
    Generates all possible full moves for the given player based on the roll result.

    Parameters:
    - player (Player): The player for whom to generate moves.
    - board (ImmutableBoard): The current state of the board.
    - roll_result (List[int]): The result of the dice roll.

    Returns:
    - List[FullMove]: A list of all possible full move sequences.
    """
    full_moves = []
    unique_boards = set()
    roll = roll_result[:]

    if roll[0] != roll[1]:
        # Sort the dice in descending order
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

        # **Apply Filtering Using Helper Function**
        full_moves = filter_full_moves_by_max_submoves(full_moves)
    else:
        # Handle double dice rolls
        handle_doubles(
            board=board,
            die_value=roll[0],  # Both dice have the same value
            full_moves=full_moves,
            unique_boards=unique_boards,
            player=player,
        )

        # **Apply Filtering Using Helper Function**
        full_moves = filter_full_moves_by_max_submoves(full_moves)

    return full_moves


def filter_full_moves_by_max_submoves(full_moves: List[FullMove]) -> List[FullMove]:
    """
    Filters the list of full moves to retain only those with the maximum number of submoves.

    Args:
        full_moves (List[FullMove]): The list of full move sequences to filter.

    Returns:
        List[FullMove]: A filtered list containing only the moves with the maximum number of submoves.
    """
    if not full_moves:
        return []

    # Determine the maximum number of submoves in any full move
    max_submoves = max(len(move.sub_move_commands) for move in full_moves)

    # Filter to retain only full moves with the maximum number of submoves
    filtered_moves = [
        move for move in full_moves if len(move.sub_move_commands) == max_submoves
    ]

    return filtered_moves
