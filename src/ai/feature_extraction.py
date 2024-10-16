# backgammon/ai/feature_extraction.py

# Import from sibling modules in the 'ai' directory
from .move_generation import get_all_possible_moves

# Import from parent directory modules
from src.board.board_class import Board
from src.players.player import Player
from src.moves.move_types import FullMove
from src.moves.handle_moves import execute_move_on_board_copy

import copy
from typing import List
import torch  # pylint: disable=import-error


def generate_all_board_features_non_batch(
    board: Board, current_player: Player, roll_result: List[int]
) -> torch.Tensor:
    """
    Generates a tensor of all possible board features based on legal moves.

    Parameters:
    - board (Board): The current board state.
    - current_player (Player): The player for whom to generate features.
    - roll_result (List[int]): The result of the dice roll.

    Returns:
    - torch.Tensor: A tensor containing feature vectors for each possible move.
    """
    legal_moves: List[FullMove] = get_all_possible_moves(
        player=current_player, board_copy=copy.copy(board), roll_result=roll_result
    )

    feature_vectors: List[List[float]] = []

    for full_move in legal_moves:
        board_copy = copy.copy(board)
        for sub_move in full_move.sub_move_commands:
            board_copy = execute_move_on_board_copy(
                board=board_copy, sub_move=sub_move, player=current_player
            )
        features = board_copy.get_board_features(current_player)
        feature_vectors.append(features)

    feature_tensor = torch.tensor(feature_vectors, dtype=torch.float32)
    return feature_tensor
