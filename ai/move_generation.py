# backgammon/ai/move_generation.py

from typing import List, Set
from copy import deepcopy
from ..board.board import Board
from ..players.player import Player
from ..moves.move_types import FullMove, SubMove
from ..moves.move_logic import get_moves_with_one_die
from ..utils.serialization import execute_move_on_board_copy, add_unique_board


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
    board = deepcopy(board_copy)
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
        # If there is more than 1 full move or 0 full moves, then execute with reversed dice
        # If there is only 1 full move and 1 submove, then don't execute to ensure that the larger die move is the only valid move.
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
