# backgammon/board/board.py
from src.players.player import Player
import torch  # pylint: disable=import-error

# Mapping from Player to tensor index
PLAYER_TO_INDEX = {
    Player.PLAYER1: 0,
    Player.PLAYER2: 1,
}


class Board:
    def __init__(self, device):
        self.device = device
        # Initialize tensors for points, bar, and borne_off on the correct device
        self.points = torch.zeros(
            (2, 24), dtype=torch.int32, device=self.device
        )  # Shape: (2, 24)
        self.bar = torch.zeros(2, dtype=torch.int32, device=self.device)  # Shape: (2,)
        self.borne_off = torch.zeros(
            2, dtype=torch.int32, device=self.device
        )  # Shape: (2,)
        self.initialize_board()

    def initialize_board(self):
        # Set up initial positions for both players
        self.set_checkers(Player.PLAYER1, 0, 2)
        self.set_checkers(Player.PLAYER1, 11, 5)
        self.set_checkers(Player.PLAYER1, 16, 3)
        self.set_checkers(Player.PLAYER1, 18, 5)

        self.set_checkers(Player.PLAYER2, 23, 2)
        self.set_checkers(Player.PLAYER2, 12, 5)
        self.set_checkers(Player.PLAYER2, 7, 3)
        self.set_checkers(Player.PLAYER2, 5, 5)

    def copy(self) -> "Board":
        new_board = Board(self.device)
        new_board.points = self.points.clone()
        new_board.bar = self.bar.clone()
        new_board.borne_off = self.borne_off.clone()
        return new_board

    def set_checkers(self, player: Player, index: int, count: int):
        player_idx = PLAYER_TO_INDEX[player]
        self.points[player_idx, index] = count

    def add_checker(self, index: int, player: Player):
        player_idx = PLAYER_TO_INDEX[player]
        opponent_idx = 1 - player_idx

        if not (0 <= index < 24 or index == -2):
            print(f"Attempted to add checker to invalid index {index}.")
            return

        if index != -2:
            opponent_checkers = self.points[opponent_idx, index]

            if opponent_checkers == 0:
                # No opponent checkers, add to own checkers
                self.points[player_idx, index] += 1
            elif opponent_checkers == 1:
                # Hit a blot
                self.points[opponent_idx, index] -= 1
                self.bar[opponent_idx] += 1
                self.points[player_idx, index] += 1
            else:
                # Blocked by opponent
                print(f"Cannot move to point {index}: blocked by opponent.")
        else:
            # Bear off
            self.borne_off[player_idx] += 1

    def remove_checker(self, index: int, player: Player):
        player_idx = PLAYER_TO_INDEX[player]

        if not (0 <= index < 24 or index == -1):
            print(f"Attempted to remove checker from invalid index {index}.")
            return

        if index != -1:
            own_checkers = self.points[player_idx, index]
            if own_checkers > 0:
                self.points[player_idx, index] -= 1
            else:
                print(
                    f"No checker to remove at point {index} for player {player.name}."
                )
        else:
            # Remove from bar
            if self.bar[player_idx] > 0:
                self.bar[player_idx] -= 1
            else:
                print("No checkers on the bar to remove.")

    def get_board_features(self, current_player: Player):
        """
        Generates a feature vector representing the board state.
        Total features: 198
        """
        features = torch.zeros(198, dtype=torch.float32, device=self.device)
        feature_index = 0

        for player_idx in [0, 1]:
            player_points = self.points[player_idx, :]  # Shape: (24,)
            for point_idx in range(24):
                checkers = player_points[point_idx].item()
                if checkers == 0:
                    features[feature_index : feature_index + 4] = torch.tensor(
                        [0.0, 0.0, 0.0, 0.0], device=self.device
                    )
                elif checkers == 1:
                    features[feature_index : feature_index + 4] = torch.tensor(
                        [1.0, 0.0, 0.0, 0.0], device=self.device
                    )
                elif checkers == 2:
                    features[feature_index : feature_index + 4] = torch.tensor(
                        [1.0, 1.0, 0.0, 0.0], device=self.device
                    )
                elif checkers >= 3:
                    features[feature_index : feature_index + 4] = torch.tensor(
                        [1.0, 1.0, 1.0, (checkers - 3.0) / 2.0], device=self.device
                    )
                else:
                    # Negative checkers should not occur
                    features[feature_index : feature_index + 4] = torch.tensor(
                        [0.0, 0.0, 0.0, 0.0], device=self.device
                    )
                feature_index += 4

            # Add bar and borne_off features
            bar_feature = self.bar[player_idx].item() / 2.0
            borne_off_feature = self.borne_off[player_idx].item() / 15.0
            features[feature_index] = bar_feature
            features[feature_index + 1] = borne_off_feature
            feature_index += 2

        # Add current player indicator
        if current_player == Player.PLAYER1:
            features[feature_index] = 1.0
            features[feature_index + 1] = 0.0
        else:
            features[feature_index] = 0.0
            features[feature_index + 1] = 1.0
        feature_index += 2

        assert (
            feature_index == 198
        ), f"Feature vector length is {feature_index}, expected 198"
        return features
