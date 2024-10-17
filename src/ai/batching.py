import torch  # pylint: disable=import-error
from typing import List
from src.board.board_class import Board
from src.players.player import Player
from src.moves.move_types import FullMove
from src.moves.get_all_moves import get_all_possible_moves

# Mapping from Player to tensor index
PLAYER_TO_INDEX = {
    Player.PLAYER1: 0,
    Player.PLAYER2: 1,
}


def generate_all_board_features(
    board: Board, current_player: Player, roll_result: List[int]
) -> torch.Tensor:
    """
    Generates a tensor of all possible board features based on legal moves,
    optimized with batch processing and tensor operations for GPU acceleration.

    Parameters:
    - board (Board): The current board state.
    - current_player (Player): The player for whom to generate features.
    - roll_result (List[int]): The result of the dice roll.

    Returns:
    - torch.Tensor: A tensor containing feature vectors for each possible move.
      Shape: (number_of_legal_moves, 198)
    """
    legal_moves: List[FullMove] = get_all_possible_moves(
        player=current_player, board_copy=board, roll_result=roll_result
    )
    if not legal_moves:
        return torch.empty((0, 198), dtype=torch.float32)

    # Apply moves and generate features in batch
    feature_tensor = apply_moves_and_get_features_in_batch(
        board, legal_moves, current_player
    )
    return feature_tensor


def apply_moves_and_get_features_in_batch(
    board: Board, legal_moves: List[FullMove], current_player: Player
) -> torch.Tensor:
    """
    Applies moves in batch and generates features using tensor operations.

    Parameters:
    - board (Board): The current board state.
    - legal_moves (List[FullMove]): A list of legal moves to apply.
    - current_player (Player): The player making the moves.

    Returns:
    - torch.Tensor: A tensor of shape (batch_size, 198) containing feature vectors.
    """
    N = len(legal_moves)
    device = board.points.device  # Get device from board

    if N == 0:
        return torch.empty((0, 198), dtype=torch.float32, device=device)

    # Initialize batched tensors for board states
    points_batch = board.points.unsqueeze(0).expand(N, -1, -1).clone()
    bar_batch = board.bar.unsqueeze(0).expand(N, -1).clone()
    borne_off_batch = board.borne_off.unsqueeze(0).expand(N, -1).clone()

    # Since M is uniform and at most 4, we can directly set M
    M = len(legal_moves[0].sub_move_commands)

    # Collect sub-move data into tensors
    start_indices = torch.empty((N, M), dtype=torch.int64, device=device)
    end_indices = torch.empty((N, M), dtype=torch.int64, device=device)
    hits_blot = torch.empty((N, M), dtype=torch.bool, device=device)
    for i, move in enumerate(legal_moves):
        for j, sub_move in enumerate(move.sub_move_commands):
            start_indices[i, j] = sub_move.start_index
            end_indices[i, j] = sub_move.end_index
            hits_blot[i, j] = sub_move.hits_blot

    # Debugging statements
    print(f"points_batch device: {points_batch.device}")
    print(f"bar_batch device: {bar_batch.device}")
    print(f"borne_off_batch device: {borne_off_batch.device}")
    print(f"start_indices device: {start_indices.device}")
    print(f"end_indices device: {end_indices.device}")
    print(f"hits_blot device: {hits_blot.device}")

    # Apply sub-moves in batch using tensor operations
    apply_sub_moves_in_batch(
        points_batch,
        bar_batch,
        borne_off_batch,
        start_indices,
        end_indices,
        hits_blot,
        current_player,
    )

    # Generate features in batch
    features_batch = get_board_features_batch_from_tensors(
        points_batch, bar_batch, borne_off_batch, current_player
    )
    return features_batch


def apply_sub_moves_in_batch(
    points_batch: torch.Tensor,
    bar_batch: torch.Tensor,
    borne_off_batch: torch.Tensor,
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
    hits_blot: torch.Tensor,
    current_player: Player,
):
    """
    Applies sub-moves in batch to the batched board tensors.

    Parameters:
    - points_batch (torch.Tensor): Batched tensor of points. Shape: (N, 2, 24)
    - bar_batch (torch.Tensor): Batched tensor of bar counts. Shape: (N, 2)
    - borne_off_batch (torch.Tensor): Batched tensor of borne off counts. Shape: (N, 2)
    - start_indices (torch.Tensor): Tensor of start indices for sub-moves. Shape: (N, M)
    - end_indices (torch.Tensor): Tensor of end indices for sub-moves. Shape: (N, M)
    - hits_blot (torch.Tensor): Tensor indicating if each sub-move hits a blot. Shape: (N, M)
    - current_player (Player): The player making the moves.
    """
    N, M = start_indices.shape
    device = points_batch.device  # Use the device of points_batch

    player_idx = PLAYER_TO_INDEX[current_player]
    opponent_idx = 1 - player_idx
    batch_indices = torch.arange(N, device=device)

    # Debugging statements
    print(f"points_batch device: {points_batch.device}")
    print(f"bar_batch device: {bar_batch.device}")
    print(f"borne_off_batch device: {borne_off_batch.device}")
    print(f"start_indices device: {start_indices.device}")
    print(f"end_indices device: {end_indices.device}")
    print(f"hits_blot device: {hits_blot.device}")
    print(f"batch_indices device: {batch_indices.device}")

    # Loop over each sub-move index (since M is small, at most 4)
    for s in range(M):
        # Extract the s-th sub-move for all moves in the batch
        start_index_s = start_indices[:, s]  # Shape: (N,)
        end_index_s = end_indices[:, s]  # Shape: (N,)
        hits_blot_s = hits_blot[:, s]  # Shape: (N,)

        # Remove checker from start index
        # If start_index == -1, it means the checker is being removed from the bar
        remove_from_bar_mask = start_index_s == -1
        remove_from_points_mask = start_index_s != -1

        # Remove checker from bar
        if remove_from_bar_mask.any():
            batch_idx_bar = batch_indices[remove_from_bar_mask]
            bar_batch[batch_idx_bar, player_idx] -= 1

        # Remove checker from points
        if remove_from_points_mask.any():
            batch_idx_points = batch_indices[remove_from_points_mask]
            start_idx_points = start_index_s[remove_from_points_mask]
            points_batch[batch_idx_points, player_idx, start_idx_points] -= 1

        # Handle hitting opponent's blot
        if hits_blot_s.any():
            hits_blot_mask = hits_blot_s
            batch_idx_hits = batch_indices[hits_blot_mask]
            end_idx_hits = end_index_s[hits_blot_mask]
            points_batch[batch_idx_hits, opponent_idx, end_idx_hits] -= 1
            bar_batch[batch_idx_hits, opponent_idx] += 1

        # Add checker to end index or bear off
        bear_off_mask = end_index_s == -2
        add_to_points_mask = end_index_s != -2

        # Add checker to points
        if add_to_points_mask.any():
            batch_idx_points_add = batch_indices[add_to_points_mask]
            end_idx_points = end_index_s[add_to_points_mask]
            points_batch[batch_idx_points_add, player_idx, end_idx_points] += 1

        # Bear off checker
        if bear_off_mask.any():
            batch_idx_bear_off = batch_indices[bear_off_mask]
            borne_off_batch[batch_idx_bear_off, player_idx] += 1


def get_board_features_batch_from_tensors(
    points_batch: torch.Tensor,
    bar_batch: torch.Tensor,
    borne_off_batch: torch.Tensor,
    current_player: Player,
) -> torch.Tensor:
    """
    Generates feature vectors for a batch of boards using tensor operations.

    Parameters:
    - points_batch (torch.Tensor): Batched tensor of points. Shape: (N, 2, 24)
    - bar_batch (torch.Tensor): Batched tensor of bar counts. Shape: (N, 2)
    - borne_off_batch (torch.Tensor): Batched tensor of borne off counts. Shape: (N, 2)
    - current_player (Player): The current player.

    Returns:
    - torch.Tensor: A tensor of shape (N, 198) containing feature vectors.
    """
    N = points_batch.shape[0]
    device = points_batch.device  # Use device from points_batch

    features = torch.zeros(N, 198, dtype=torch.float32, device=device)
    feature_index = 0

    # Debugging statements
    print(f"points_batch device: {points_batch.device}")
    print(f"bar_batch device: {bar_batch.device}")
    print(f"borne_off_batch device: {borne_off_batch.device}")
    print(f"features device: {features.device}")

    for player_idx in [0, 1]:
        player_points = points_batch[:, player_idx, :]  # Shape: (N, 24)
        for point_idx in range(24):
            checkers = player_points[:, point_idx]  # Shape: (N,)

            features_slice = torch.zeros(N, 4, dtype=torch.float32, device=device)

            # Checkers == 1
            mask_one = checkers == 1
            features_slice[mask_one, 0] = 1.0

            # Checkers == 2
            mask_two = checkers == 2
            features_slice[mask_two, 0:2] = 1.0

            # Checkers >= 3
            mask_three_or_more = checkers >= 3
            features_slice[mask_three_or_more, 0:3] = 1.0
            features_slice[mask_three_or_more, 3] = (
                checkers[mask_three_or_more].float() - 3.0
            ) / 2.0

            # Assign to the features tensor
            features[:, feature_index : feature_index + 4] = features_slice
            feature_index += 4

        # Add bar and borne_off features
        bar_feature = bar_batch[:, player_idx].float() / 2.0  # Shape: (N,)
        borne_off_feature = borne_off_batch[:, player_idx].float() / 15.0  # Shape: (N,)

        features[:, feature_index] = bar_feature
        features[:, feature_index + 1] = borne_off_feature
        feature_index += 2

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
