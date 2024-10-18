from dataclasses import dataclass
from enum import IntEnum
from typing import List
import torch
from src.moves.move_types import SubMove, Position, FullMove
from src.players.player import Player
import logging

logging.basicConfig(
    level=logging.WARNING,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImmutableBoard:
    tensor: torch.Tensor  # Shape: (4, 24), dtype: torch.int8

    @staticmethod
    def initial_board(device: torch.device) -> "ImmutableBoard":
        tensor = torch.zeros((4, 24), dtype=torch.int8, device=device)
        # Set initial positions for both players using Position enum
        # Player 1
        tensor[Player.PLAYER1.value, Position.P_0.value] = 2
        tensor[Player.PLAYER1.value, Position.P_11.value] = 5
        tensor[Player.PLAYER1.value, Position.P_16.value] = 3
        tensor[Player.PLAYER1.value, Position.P_18.value] = 5
        # Player 2
        tensor[Player.PLAYER2.value, Position.P_23.value] = 2
        tensor[Player.PLAYER2.value, Position.P_12.value] = 5
        tensor[Player.PLAYER2.value, Position.P_7.value] = 3
        tensor[Player.PLAYER2.value, Position.P_5.value] = 5
        # Bar and borne-off checkers are zeros
        return ImmutableBoard(tensor)

    def move_checker(self, player: Player, sub_move: SubMove) -> "ImmutableBoard":
        """
        Returns a new Board with the sub-move applied.
        """
        new_tensor = self.tensor.clone()
        player_idx = player.value  # 0 or 1
        opponent_idx = 1 - player_idx

        start = sub_move.start
        end = sub_move.end
        hits_blot = sub_move.hits_blot

        # Remove checker from start
        if start == Position.BAR:
            if new_tensor[2, player_idx] > 0:
                new_tensor[2, player_idx] -= 1
            else:
                logger.warning(
                    f"No checker to remove from bar for player {player.name}"
                )
                return self  # Return the original board unchanged
        else:
            if new_tensor[player_idx, start.value] > 0:
                new_tensor[player_idx, start.value] -= 1
            else:
                logger.warning(
                    f"No checker to remove at point {start.name} for player {player.name}"
                )
                return self  # Return the original board unchanged

        # Handle hitting a blot
        if hits_blot:
            if new_tensor[opponent_idx, end.value] > 0:
                new_tensor[opponent_idx, end.value] -= 1
                new_tensor[2, opponent_idx] += 1  # Add to opponent's bar
            else:
                logger.warning(
                    f"No blot to hit at point {end.name} for player {player.name}"
                )
                return self  # Return the original board unchanged

        # Add checker to end
        if end == Position.BEAR_OFF:
            new_tensor[3, player_idx] += 1
        else:
            new_tensor[player_idx, end.value] += 1

        return ImmutableBoard(new_tensor)

    def add_checker(self, position: Position, player: Player) -> "ImmutableBoard":
        """
        Adds a checker to the specified position for the given player.
        Returns a new Board instance with the checker added.
        """
        new_tensor = self.tensor.clone()
        player_idx = player.value
        opponent_idx = 1 - player_idx

        if position == Position.BEAR_OFF:
            # Adding to bear-off
            new_tensor[3, player_idx] += 1
        elif position == Position.BAR:
            # Adding to the bar
            new_tensor[2, player_idx] += 1
        elif Position.P_0 <= position <= Position.P_23:
            # Adding to a specific point
            opponent_checkers = new_tensor[opponent_idx, position.value].item()

            if opponent_checkers == 0:
                # No opponent checkers, add to own checkers
                new_tensor[player_idx, position.value] += 1
            elif opponent_checkers == 1:
                # Hit a blot
                new_tensor[opponent_idx, position.value] -= 1
                new_tensor[2, opponent_idx] += 1  # Add to opponent's bar
                new_tensor[player_idx, position.value] += 1
            else:
                # Blocked by opponent
                logger.warning(
                    f"Cannot add checker to {position.name}: blocked by opponent."
                )
                return self  # Return the original board unchanged
        else:
            logger.warning(f"Invalid position {position} for adding a checker.")
            return self  # Return the original board unchanged

        return ImmutableBoard(new_tensor)

    def remove_checker(self, position: Position, player: Player) -> "ImmutableBoard":
        """
        Removes a checker from the specified position for the given player.
        Returns a new Board instance with the checker removed.
        """
        new_tensor = self.tensor.clone()
        player_idx = player.value

        if position == Position.BAR:
            # Removing from the bar
            if new_tensor[2, player_idx] > 0:
                new_tensor[2, player_idx] -= 1
            else:
                logger.warning(
                    f"No checkers on the bar to remove for player {player.name}"
                )
                return self  # Return the original board unchanged
        elif Position.P_0 <= position <= Position.P_23:
            # Removing from a specific point
            if new_tensor[player_idx, position.value] > 0:
                new_tensor[player_idx, position.value] -= 1
            else:
                logger.warning(
                    f"No checker to remove at point {position.name} for player {player.name}"
                )
                return self  # Return the original board unchanged
        elif position == Position.BEAR_OFF:
            # Removing from bear-off
            if new_tensor[3, player_idx] > 0:
                new_tensor[3, player_idx] -= 1
            else:
                logger.warning(
                    f"No checkers to remove from bear-off for player {player.name}"
                )
                return self  # Return the original board unchanged
        else:
            logger.warning(f"Invalid position {position} for removing a checker.")
            return self  # Return the original board unchanged

        return ImmutableBoard(new_tensor)

    def get_board_features(self, current_player: Player) -> torch.Tensor:
        features = torch.zeros(198, dtype=torch.float32, device=self.tensor.device)
        feature_index = 0
        # Iterate over players
        for player_idx in [Player.PLAYER1.value, Player.PLAYER2.value]:
            player_points = self.tensor[player_idx, :]  # Shape: (24,)
            for point in Position:
                if point.value >= 24:
                    continue  # Skip BAR and BEAR_OFF
                checkers = player_points[point.value].item()
                features_slice = torch.zeros(
                    4, dtype=torch.float32, device=self.tensor.device
                )
                if checkers == 1:
                    features_slice[0] = 1.0
                elif checkers == 2:
                    features_slice[0:2] = 1.0
                elif checkers >= 3:
                    features_slice[0:3] = 1.0
                    features_slice[3] = (float(checkers - 3)) / 2.0
                features[feature_index : feature_index + 4] = features_slice
                feature_index += 4

            # Add bar and borne-off features
            bar_checkers = self.tensor[2, player_idx].item()
            borne_off_checkers = self.tensor[3, player_idx].item()
            features[feature_index] = float(bar_checkers) / 2.0
            features[feature_index + 1] = float(borne_off_checkers) / 15.0
            feature_index += 2

        # Add current player indicator
        if current_player == Player.PLAYER1:
            features[feature_index] = 1.0
            features[feature_index + 1] = 0.0
        else:
            features[feature_index] = 0.0
            features[feature_index + 1] = 1.0
        feature_index += 2

        if feature_index != 198:
            logger.warning(f"Feature vector length is {feature_index}, expected 198")
        return features


def execute_sub_move_on_board(
    board: ImmutableBoard, sub_move: SubMove, player: Player
) -> ImmutableBoard:
    """
    Executes a sub-move on the board and returns a new Board instance.
    """
    return board.move_checker(player, sub_move)


def execute_full_move_on_board_copy(
    board: ImmutableBoard, full_move: FullMove
) -> ImmutableBoard:
    """
    Executes a full move (composed of sub-moves) on a copy of the board.
    """
    new_board = board
    for sub_move in full_move.sub_move_commands:
        new_board = execute_sub_move_on_board(new_board, sub_move, full_move.player)
    return new_board


def board_hash(board: ImmutableBoard) -> int:
    """
    Generates a hashable representation of the board.
    """
    try:
        # Attempt to hash without moving to CPU by using a GPU-compatible hashing method
        # Note: Standard Python hash functions require data on CPU, so this is a workaround
        return hash(board.tensor.detach().cpu().numpy().tobytes())
    except Exception as e:
        logger.warning(f"Failed to hash board tensor: {e}")
        return 0  # Return a default hash value in case of failure
