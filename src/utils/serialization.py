# backgammon/utils/serialization.py

import pickle
from typing import List, Set
from src.board.board_class import Board
from src.players.player import Player
from src.moves.move_types import SubMove, FullMove

# todo: add process pool to execute move on board copy.  they do not have to be unique (they already are unique)

PLAYER_TO_INDEX = {
    Player.PLAYER1: 0,
    Player.PLAYER2: 1,
}


def execute_move_on_board_copy(
    board: Board, sub_move: SubMove, player: Player
) -> Board:
    """
    Executes a sub-move on a copy of the board.

    Parameters:
    - board (Board): The current board state.
    - sub_move (SubMove): The sub-move to execute.
    - player (Player): The player making the move.

    Returns:
    - Board: The new board state after applying the move.
    """
    board_copy = board.copy()
    move = sub_move
    player_idx = PLAYER_TO_INDEX[player]
    opponent_idx = 1 - player_idx

    # Remove the checker from the start index
    board_copy.remove_checker(move.start_index, player)

    if move.hits_blot:
        # Remove opponent's checker from the end index and place it on the bar
        board_copy.points[opponent_idx, move.end_index] -= 1
        board_copy.bar[opponent_idx] += 1

    # Add the checker to the end index or bear off
    if move.end_index == -2:
        # Bear off
        board_copy.borne_off[player_idx] += 1
    else:
        board_copy.add_checker(move.end_index, player)

    return board_copy


def board_hash(board: Board) -> int:
    """
    Generates a hashable representation of the board.
    """
    return hash(
        (
            tuple(board.points.view(-1).tolist()),
            tuple(board.bar.tolist()),
            tuple(board.borne_off.tolist()),
        )
    )


def add_unique_board(
    board: Board,
    moves: List[SubMove],
    full_moves: List[FullMove],
    unique_boards: Set[int],
    player: Player,
):
    board_h = board_hash(board)
    if board_h not in unique_boards:
        unique_boards.add(board_h)
        full_move = FullMove(sub_move_commands=moves.copy(), player=player)
        full_moves.append(full_move)


def process_board(board: Board, player: Player):
    """
    For testing, perform serialization and return the number of checkers borne off by the player.

    Parameters:
    - board (Board): The board to process.
    - player (Player): The player whose borne off checkers to count.

    Returns:
    - int: Number of checkers borne off by the player.
    """
    serialized = pickle.dumps(board)
    deserialized = pickle.loads(serialized)
    return deserialized.borne_off[player]
