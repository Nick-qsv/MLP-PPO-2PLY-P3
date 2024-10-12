# backgammon/ai/feature_extraction.py

import torch
from typing import List
from ..board.board import Board
from ..players.player import Player
from ..moves.move_types import FullMove
from .move_generation import get_all_possible_moves
from ..utils.serialization import execute_move_on_board_copy


def generate_all_board_features(
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


def get_move_features(full_move: FullMove) -> List[float]:
    """
    Converts a FullMove into a normalized feature vector.

    Parameters:
    - full_move (FullMove): The full move to convert.

    Returns:
    - List[float]: The feature vector representing the move.
    """
    features_vector = []
    max_submoves = 4  # Maximum number of sub-moves in a full move

    for submove in full_move.sub_move_commands:
        # Normalize start_index (-1 to 23) to 0.0 to 1.0
        start_index_normalized = (submove.start_index + 1) / 24.0

        # Normalize end_index (-2 to 23) to 0.0 to 1.0
        end_index_normalized = (submove.end_index + 2) / 25.0

        features_vector += [
            start_index_normalized,
            end_index_normalized,
            1.0 if submove.hits_blot else 0.0,
        ]

    # Pad the remaining sub-move slots with special values
    num_padding = max_submoves - len(full_move.sub_move_commands)
    for _ in range(num_padding):
        features_vector += [-1.0, -1.0, -1.0]  # Indicates no move

    assert len(features_vector) == max_submoves * 3, "Feature vector length mismatch"
    return features_vector
