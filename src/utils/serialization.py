# backgammon/utils/serialization.py

import pickle
from typing import List, Set
from src.board.board import Board
from src.players.player import Player
from src.moves.move_types import SubMove, FullMove
from src.board.point_state import PointState
import copy

# todo: add process pool to execute move on board copy.  they do not have to be unique (they already are unique)


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
    board_copy = copy.deepcopy(board)
    move = sub_move

    board_copy.remove_checker(move.start_index, player)

    if move.hits_blot:
        opponent = Player.PLAYER2 if player == Player.PLAYER1 else Player.PLAYER1
        point_state = board_copy.points[move.end_index]
        if point_state != PointState.EMPTY and point_state[0] == PointState.OWNED:
            owned_state = point_state[1]
            if owned_state.player == opponent and owned_state.count == 1:
                board_copy.points[move.end_index] = PointState.EMPTY
                board_copy.bar[opponent] += 1
            else:
                print("Expected to hit a blot, but conditions were not met.")
        else:
            print("Expected to hit a blot, but conditions were not met.")

    if move.end_index == -2:
        board_copy.borne_off[player] += 1
    else:
        board_copy.add_checker(move.end_index, player)

    return board_copy


def add_unique_board(
    board: Board,
    moves: List[SubMove],
    full_moves: List[FullMove],
    unique_boards: Set[Board],
    player: Player,
):
    """
    Adds a unique board state to the set of unique boards and appends the corresponding full move.

    Parameters:
    - board (Board): The new board state.
    - moves (List[SubMove]): The sequence of sub-moves leading to this state.
    - full_moves (List[FullMove]): The list to append the new full move to.
    - unique_boards (Set[Board]): The set of unique board states.
    - player (Player): The player making the move.
    """
    if board not in unique_boards:
        unique_boards.add(board)
        full_move = FullMove(sub_move_commands=moves, player=player)
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
