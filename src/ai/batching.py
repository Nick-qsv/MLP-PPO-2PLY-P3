import torch  # pylint: disable=import-error
from typing import List
from src.board.board_class import Board
from src.players.player import Player
from src.moves.move_types import FullMove
from src.moves.get_all_moves import get_all_possible_moves


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
        player=current_player, board_copy=board, roll_result=roll_result
    )

    # Collect all board copies after applying moves
    board_copies = apply_moves_in_batch(board, legal_moves, current_player)

    # Generate features in batch
    feature_tensor = get_board_features_batch(board_copies, current_player)
    return feature_tensor


def apply_moves_in_batch(
    board: Board, legal_moves: List[FullMove], player: Player
) -> List[Board]:
    """
    Applies a batch of moves to a board and returns a list of resulting boards.

    Parameters:
    - board (Board): The current board state.
    - legal_moves (List[FullMove]): A list of legal moves to apply.
    - player (Player): The player making the moves.

    Returns:
    - List[Board]: A list of new board states after applying each move.
    """
    board_copies = []
    for full_move in legal_moves:
        board_copy = board.copy()
        for sub_move in full_move.sub_move_commands:
            board_copy.apply_sub_move(sub_move, player)
        board_copies.append(board_copy)
    return board_copies


def get_board_features_batch(
    boards: List[Board], current_player: Player
) -> torch.Tensor:
    """
    Generates a tensor of board features for a batch of boards.

    Parameters:
    - boards (List[Board]): A list of board states.
    - current_player (Player): The player for whom to generate features.

    Returns:
    - torch.Tensor: A tensor of shape (batch_size, 198).
    """
    feature_vectors = [board.get_board_features(current_player) for board in boards]
    features = torch.tensor(feature_vectors, dtype=torch.float32)
    return features
