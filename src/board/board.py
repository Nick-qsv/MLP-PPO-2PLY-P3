# backgammon/board/board.py

from dataclasses import dataclass, field
from typing import List, Dict
from .point_state import PointState
from .owned_state import OwnedState
from src.players.player import Player


@dataclass
class Board:
    points: List = field(default_factory=lambda: [PointState.EMPTY] * 24)
    bar: Dict[Player, int] = field(
        default_factory=lambda: {Player.PLAYER1: 0, Player.PLAYER2: 0}
    )
    borne_off: Dict[Player, int] = field(
        default_factory=lambda: {Player.PLAYER1: 0, Player.PLAYER2: 0}
    )

    def __post_init__(self):
        self.set_checkers(Player.PLAYER1, 0, 2)
        self.set_checkers(Player.PLAYER1, 11, 5)
        self.set_checkers(Player.PLAYER1, 16, 3)
        self.set_checkers(Player.PLAYER1, 18, 5)

        self.set_checkers(Player.PLAYER2, 23, 2)
        self.set_checkers(Player.PLAYER2, 12, 5)
        self.set_checkers(Player.PLAYER2, 7, 3)
        self.set_checkers(Player.PLAYER2, 5, 5)

    def set_checkers(self, player: Player, index: int, count: int):
        self.points[index] = (PointState.OWNED, OwnedState(count, player))

    def add_checker(self, index: int, player: Player):
        if not (0 <= index < 24 or index == -2):
            print(f"Attempted to add checker to invalid index {index}.")
            return
        if index != -2:
            point_state = self.points[index]
            if point_state == PointState.EMPTY:
                self.points[index] = (PointState.OWNED, OwnedState(1, player))
            elif point_state[0] == PointState.OWNED:
                owned_state = point_state[1]
                if owned_state.player == player:
                    self.points[index] = (
                        PointState.OWNED,
                        OwnedState(owned_state.count + 1, player),
                    )
                elif owned_state.count == 1:
                    # Hit a blot
                    self.bar[owned_state.player] += 1
                    self.points[index] = (PointState.OWNED, OwnedState(1, player))
                else:
                    print(f"Cannot move to point {index}: blocked by opponent.")
        else:
            self.borne_off[player] += 1

    def remove_checker(self, index: int, player: Player):
        if not (0 <= index < 24 or index == -1):
            print(f"Attempted to remove checker from invalid index {index}.")
            return
        if index != -1:
            point_state = self.points[index]
            if point_state == PointState.EMPTY:
                print(f"No checker to remove at point {index}.")
            elif point_state[0] == PointState.OWNED:
                owned_state = point_state[1]
                if owned_state.player != player:
                    print(
                        f"Cannot remove checker: point {index} is owned by the opponent."
                    )
                    return
                if owned_state.count > 1:
                    self.points[index] = (
                        PointState.OWNED,
                        OwnedState(owned_state.count - 1, player),
                    )
                else:
                    self.points[index] = PointState.EMPTY
        else:
            if self.bar[player] > 0:
                self.bar[player] -= 1
            else:
                print("No checkers on the bar to remove.")

    def __eq__(self, other):
        return (
            self.points == other.points
            and self.bar == other.bar
            and self.borne_off == other.borne_off
        )

    def __hash__(self):
        return (
            hash(tuple(self.points))
            ^ hash(frozenset(self.bar.items()))
            ^ hash(frozenset(self.borne_off.items()))
        )

    def get_board_features(self, current_player: Player) -> List[float]:
        """
        - encode each point (24) with 4 units => 4 * 24 = 96
        - for each player => 96 * 2 = 192
        - 2 units indicating who is the current player
        - 2 units for white and black bar checkers
        - 2 units for white and black off checkers
        - tot = 192 + 2 + 2 + 2 = 198
        """
        features_vector = []
        for player in [Player.PLAYER1, Player.PLAYER2]:
            for point in self.points:
                if point == PointState.EMPTY:
                    features_vector += [0.0, 0.0, 0.0, 0.0]
                else:
                    owned_state = point[1]
                    if owned_state.player == player:
                        checkers = owned_state.count
                        if checkers == 1:
                            features_vector += [1.0, 0.0, 0.0, 0.0]
                        elif checkers == 2:
                            features_vector += [1.0, 1.0, 0.0, 0.0]
                        elif checkers >= 3:
                            features_vector += [1.0, 1.0, 1.0, (checkers - 3.0) / 2.0]
                    else:
                        features_vector += [0.0, 0.0, 0.0, 0.0]

            features_vector += [self.bar[player] / 2.0, self.borne_off[player] / 15.0]

        if current_player == Player.PLAYER1:
            features_vector += [1.0, 0.0]
        else:
            features_vector += [0.0, 1.0]

        assert (
            len(features_vector) == 198
        ), f"Should be 198 instead of {len(features_vector)}"
        return features_vector

    def print_board(self):
        """
        Prints the board representation.
        """
        board_representation = []
        for i, point in enumerate(self.points):
            if point == PointState.EMPTY:
                board_representation.append(f"Point {i}: Empty")
            else:
                owned_state = point[1]
                board_representation.append(
                    f"Point {i}: {owned_state.count} checkers by {owned_state.player.name}"
                )
        print("\n".join(board_representation))

        print("\nBar checkers:")
        print(f"Player 1: {self.bar[Player.PLAYER1]}")
        print(f"Player 2: {self.bar[Player.PLAYER2]}")

        print("\nBorne off checkers:")
        print(f"Player 1: {self.borne_off[Player.PLAYER1]}")
        print(f"Player 2: {self.borne_off[Player.PLAYER2]}")

    def test_get_board_features(self, current_player: Player):
        """
        Prints the board and its feature representation.
        """
        print("Board Representation:")
        self.print_board()

        features = self.get_board_features(current_player)

        print("\nBoard Features Representation:")
        feature_index = 0
        for player in [Player.PLAYER1, Player.PLAYER2]:
            print(f"\nFeatures for {player.name}:")
            for point in range(24):
                feature_slice = features[feature_index : feature_index + 4]
                print(f"Point {point + 1} ({player.name}): {feature_slice}")
                feature_index += 4
            bar_feature = features[feature_index]
            borne_off_feature = features[feature_index + 1]
            print(f"Bar checkers ({player.name}): {bar_feature}")
            print(f"Borne off checkers ({player.name}): {borne_off_feature}")
            feature_index += 2

        current_player_feature = features[feature_index : feature_index + 2]
        print(f"\nCurrent player feature: {current_player_feature}")
