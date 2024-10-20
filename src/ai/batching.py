import torch
from typing import List
from src.players.player import Player
from src.moves.move_types import FullMove, Position
from src.board.immutable_board import ImmutableBoard, execute_full_move_on_board_copy
from src.board.immutable_board import board_to_string
from src.utils.decorators import profile, profiling_data


@profile
def generate_all_board_features(
    board: ImmutableBoard,
    current_player: Player,
    legal_moves: List[FullMove],
    roll_result: List[int],
) -> torch.Tensor:
    """
    Generates a tensor of all possible board features based on legal moves,
    optimized with batch processing and tensor operations for GPU acceleration.

    Parameters:
    - board (ImmutableBoard): The current board state.
    - current_player (Player): The player for whom to generate features.
    - legal_moves (List[FullMove]): List of possible full moves.
    - roll_result (List[int]): The result of the dice roll.
    - board_str (str): String representation of the board.

    Returns:
    - torch.Tensor: A tensor containing feature vectors for each possible move.
      Shape: (number_of_legal_moves, 198)
    """
    if not legal_moves:
        return torch.empty((0, 198), dtype=torch.float32, device=board.tensor.device)

    # Check for consistent M
    M_list = [len(move.sub_move_commands) for move in legal_moves]
    M = M_list[0]
    if not all(m == M for m in M_list):
        print("\nInconsistent M in batch detected.")
        print(f"Roll Result: {roll_result}")
        print("Board State:")
        print(board_to_string(board))

        # Print available moves
        print("\nAvailable moves:")
        for i, move in enumerate(legal_moves):
            # Generate the description for each SubMove
            moves_description = ", ".join(
                f"[{'bar' if sub_move.start == Position.BAR else sub_move.start.name}, "
                f"{'off' if sub_move.end == Position.BEAR_OFF else sub_move.end.name}, "
                f"{'*' if sub_move.hits_blot else '-'}]"
                for sub_move in move.sub_move_commands
            )

            # Format the full move string
            full_move_str = f"Full Move ({move.player.name}): {moves_description}"

            # Print the formatted move
            print(f"Move {i}: {full_move_str}")

        # Raise an error or handle accordingly
        raise ValueError("Inconsistent number of SubMoves (M) in batch.")

    # Apply moves to get resulting boards
    new_boards = [execute_full_move_on_board_copy(board, move) for move in legal_moves]

    # Stack the board tensors into a batch tensor
    # Each board tensor is of shape (4, 24), so batch tensor will be (N, 4, 24)
    board_tensors = torch.stack([b.tensor for b in new_boards], dim=0)

    # Generate features in batch
    features_batch = get_board_features_batch_from_tensors(
        board_tensors, current_player
    )
    return features_batch


@profile
def get_board_features_batch_from_tensors(
    board_tensors: torch.Tensor, current_player: Player
) -> torch.Tensor:
    """
    Generates feature vectors for a batch of boards using tensor operations.

    Parameters:
    - board_tensors (torch.Tensor): Batched tensor of boards. Shape: (N, 4, 24)
    - current_player (Player): The current player.

    Returns:
    - torch.Tensor: A tensor of shape (N, 198) containing feature vectors.
    """
    N = board_tensors.shape[0]
    device = board_tensors.device

    features = torch.zeros(N, 198, dtype=torch.float32, device=device)
    feature_index = 0

    for player_idx in [Player.PLAYER1.value, Player.PLAYER2.value]:
        # Get player points: (N, 24)
        player_points = board_tensors[:, player_idx, :]  # Shape: (N, 24)
        for point_idx in range(24):
            checkers = player_points[:, point_idx]  # Shape: (N,)
            features_slice = torch.zeros(N, 4, dtype=torch.float32, device=device)

            # Compute features based on number of checkers
            # For checkers == 1
            mask_one = checkers == 1
            features_slice[mask_one, 0] = 1.0

            # For checkers == 2
            mask_two = checkers == 2
            features_slice[mask_two, 0:2] = 1.0

            # For checkers >= 3
            mask_three_or_more = checkers >= 3
            features_slice[mask_three_or_more, 0:3] = 1.0
            features_slice[mask_three_or_more, 3] = (
                checkers[mask_three_or_more].float() - 3.0
            ) / 2.0

            # Assign to features
            features[:, feature_index : feature_index + 4] = features_slice
            feature_index += 4

        # Bar checkers
        bar_checkers = board_tensors[:, 2, player_idx]  # Shape: (N,)
        features[:, feature_index] = bar_checkers.float() / 2.0
        feature_index += 1

        # Borne off checkers
        borne_off_checkers = board_tensors[:, 3, player_idx]  # Shape: (N,)
        features[:, feature_index] = borne_off_checkers.float() / 15.0
        feature_index += 1

    # Add current player indicator
    if current_player == Player.PLAYER1:
        features[:, feature_index] = 1.0
        features[:, feature_index + 1] = 0.0
    else:
        features[:, feature_index] = 0.0
        features[:, feature_index + 1] = 1.0
    feature_index += 2

    assert (
        feature_index == 198
    ), f"Feature vector length is {feature_index}, expected 198"
    return features
