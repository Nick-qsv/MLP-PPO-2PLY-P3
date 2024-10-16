from typing import List
from src.board import Board
from src.moves.move_types import FullMove
from src.players import Player
from src.moves.handle_moves import handle_non_doubles, handle_doubles


def get_all_possible_moves(
    player: Player, board_copy: Board, roll_result: List[int]
) -> List[FullMove]:
    """
    Generates all possible full moves for the given player based on the roll result.

    Parameters:
    - player (Player): The player for whom to generate moves.
    - board_copy (Board): A copy of the current board state.
    - roll_result (List[int]): The result of the dice roll.

    Returns:
    - List[FullMove]: A list of all possible full moves.
    """
    full_moves = []
    unique_boards = set()
    board = board_copy.copy()
    roll = roll_result[:]

    if roll[0] != roll[1]:
        roll.sort(reverse=True)
        handle_non_doubles(
            board=board,
            roll=roll,
            full_moves=full_moves,
            unique_boards=unique_boards,
            player=player,
        )
        if not full_moves or not (
            len(full_moves) == 1 and len(full_moves[0].sub_move_commands) == 1
        ):
            handle_non_doubles(
                board=board,
                roll=roll,
                full_moves=full_moves,
                unique_boards=unique_boards,
                player=player,
                reverse=True,
            )
    else:
        handle_doubles(
            board=board,
            die_value=roll[0],
            full_moves=full_moves,
            unique_boards=unique_boards,
            player=player,
        )

    return full_moves
