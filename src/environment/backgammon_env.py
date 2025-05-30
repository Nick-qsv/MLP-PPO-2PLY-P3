import gym
import numpy as np
import torch
from gym import spaces
from src.players.player import Player
from src.moves.get_all_moves import get_all_possible_moves
from src.ai.batching import generate_all_board_features
from src.board.immutable_board import (
    ImmutableBoard,
    execute_full_move_on_board_copy,
)
from src.moves.move_types import Position
from typing import Dict

TOKEN = {
    Player.PLAYER1: "●",  # Example token for Player 1
    Player.PLAYER2: "○",  # Example token for Player 2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reward Variables (Configurable)
REWARD_INVALID_ACTION = -1.0
REWARD_PASS = 0.0
REWARD_HIT = 0.01
REWARD_WIN_NORMAL = 1.0
REWARD_WIN_GAMMON = 1.5
REWARD_WIN_BACKGAMMON = 2.0


def get_opponent(player: Player) -> Player:
    return Player.PLAYER2 if player == Player.PLAYER1 else Player.PLAYER1


class BackgammonEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, match_length=15, max_legal_moves=500, device=None):
        super(BackgammonEnv, self).__init__()

        self.match_length = match_length
        self.player_scores: Dict[Player, int] = {
            Player.PLAYER1: 0,
            Player.PLAYER2: 0,
        }
        self.current_match_winner = None

        # Set the device
        self.device = device if device is not None else torch.device("cpu")

        self.board = ImmutableBoard.initial_board(device=self.device)
        self.current_player = Player.PLAYER1
        self.game_over = False
        self.match_over = False

        self.max_legal_moves = max_legal_moves

        # Observation space
        board_feature_length = 198  # From get_board_features
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(board_feature_length,),
            dtype=np.float32,
        )

        # Action space
        self.action_space = spaces.Discrete(self.max_legal_moves)

        # Variables for dice roll and legal moves
        self.roll_result = None
        self.action_mask = torch.zeros(
            self.max_legal_moves, dtype=torch.float32, device=self.device
        )
        self.legal_board_features = None  # Tensor of possible next board features
        self.legal_moves = []  # List of FullMove objects

    def reset(self):
        if self.match_over:
            self.player_scores = {Player.PLAYER1: 0, Player.PLAYER2: 0}
            self.match_over = False
            self.current_match_winner = None

        # Reset the board
        self.board = ImmutableBoard.initial_board(device=self.device)
        self.game_over = False

        # Alternate starting player
        self.current_player = (
            Player.PLAYER1 if self.current_player == Player.PLAYER2 else Player.PLAYER2
        )

        # Roll dice to determine who starts
        self.roll_dice()
        while self.roll_result[0] == self.roll_result[1]:
            self.roll_dice()

        # The player with the higher roll starts
        if self.roll_result[0] < self.roll_result[1]:
            self.current_player = Player.PLAYER2
        else:
            self.current_player = Player.PLAYER1

        # Roll dice for the first move, ensuring it's not doubles
        self.roll_dice()
        while self.roll_result[0] == self.roll_result[1]:
            self.roll_dice()

        # Update legal moves and board features based on the first non-doubles roll
        self.update_legal_moves()

        observation = self.get_observation()
        return observation

    def step(self, action):
        # Initialize the info dictionary with current_player
        info = {"current_player": self.current_player}

        if self.game_over:
            observation = self.reset()
            return observation, torch.tensor(0.0, device=self.device), True, info

        # Check if there are any legal actions
        if self.action_mask.sum() == 0:
            # No legal actions, pass the turn to the next player
            reward = torch.tensor(REWARD_PASS, device=self.device)
            done = False
            # Pass the turn
            self.pass_turn()
            self.roll_dice()
            self.update_legal_moves()

            # Get the new observation after passing the turn
            observation = self.get_observation()
            return (
                observation,
                reward,
                done,
                {**info, "info": "No legal actions, turn passed"},
            )

        # Validate action
        if not self.action_mask[action]:
            # Invalid action selected
            reward = torch.tensor(REWARD_INVALID_ACTION, device=self.device)
            done = False
            print(f"Invalid action selected: {action}. Assigned reward: {reward}")
            observation = self.get_observation()
            return observation, reward, done, {**info, "info": "Invalid action"}

        # Execute the Selected Move by applying the corresponding FullMove
        selected_move = self.legal_moves[action]
        self.board = execute_full_move_on_board_copy(self.board, selected_move)

        # Check for game over
        if self.board.tensor[3, self.current_player.value].item() == 15:
            # Winning conditions
            is_backgammon = self.check_for_backgammon(self.current_player)
            is_gammon = False
            if not is_backgammon:
                is_gammon = self.check_for_gammon(self.current_player)

            if is_backgammon:
                game_score = 3
                reward = torch.tensor(REWARD_WIN_BACKGAMMON, device=self.device)
            elif is_gammon:
                game_score = 2
                reward = torch.tensor(REWARD_WIN_GAMMON, device=self.device)
            else:
                game_score = 1
                reward = torch.tensor(REWARD_WIN_NORMAL, device=self.device)
            info.update({"winner": self.current_player, "game_score": game_score})
            self.player_scores[self.current_player] += game_score
            self.game_over = True
            done = True

            # Check if match is over
            if self.player_scores[self.current_player] >= self.match_length:
                # Match is over
                self.current_match_winner = self.current_player
                self.match_over = True
        else:
            reward = torch.tensor(0.0, device=self.device)
            done = False
            # Pass the turn to the other player
            self.pass_turn()
            self.roll_dice()
            self.update_legal_moves()

        observation = self.get_observation()
        return observation, reward, done, info

    def get_observation(self):
        # Board features
        board_features = self.board.get_board_features(self.current_player)
        return board_features  # Return tensor on self.device

    def update_legal_moves(self):
        # Generate legal moves
        self.legal_moves = get_all_possible_moves(
            player=self.current_player,
            board=self.board,
            roll_result=self.roll_result,
        )

        # Generate legal board features for the action mask
        self.legal_board_features = (
            generate_all_board_features(
                board=self.board,
                current_player=self.current_player,
                legal_moves=self.legal_moves,
                roll_result=self.roll_result,
            )
            if self.legal_moves
            else torch.empty((0, 198), dtype=torch.float32, device=self.device)
        )

        num_moves = self.legal_board_features.size(0)
        if num_moves > self.max_legal_moves:
            self.legal_board_features = self.legal_board_features[
                : self.max_legal_moves, :
            ]
            self.legal_moves = self.legal_moves[: self.max_legal_moves]

        num_moves = self.legal_board_features.size(0)

        # Update action_mask
        self.action_mask = torch.zeros(
            self.max_legal_moves, dtype=torch.float32, device=self.device
        )
        self.action_mask[:num_moves] = 1.0

        # If there are fewer moves than max_legal_moves, pad the features
        if num_moves < self.max_legal_moves:
            padding_length = self.max_legal_moves - num_moves
            padding = torch.zeros(
                (padding_length, self.legal_board_features.size(1)),
                dtype=self.legal_board_features.dtype,
                device=self.device,
            )
            self.legal_board_features = torch.cat(
                [self.legal_board_features, padding], dim=0
            )

    def roll_dice(self):
        self.roll_result = [np.random.randint(1, 7), np.random.randint(1, 7)]

    def pass_turn(self):
        self.current_player = (
            Player.PLAYER2 if self.current_player == Player.PLAYER1 else Player.PLAYER1
        )

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Only 'human' mode is supported")

        points = []
        colors = []
        for point_idx in range(24):
            player1_checkers = self.board.tensor[Player.PLAYER1.value, point_idx].item()
            player2_checkers = self.board.tensor[Player.PLAYER2.value, point_idx].item()

            if player1_checkers > 0 and player2_checkers > 0:
                # This should not happen in Backgammon. Handle as an error or decide on precedence.
                print(
                    f"Invalid board state at point {point_idx}: Both players have checkers."
                )
                points.append(0)
                colors.append("?")
            elif player1_checkers > 0:
                points.append(player1_checkers)
                colors.append(TOKEN.get(Player.PLAYER1, "?"))
            elif player2_checkers > 0:
                points.append(player2_checkers)
                colors.append(TOKEN.get(Player.PLAYER2, "?"))
            else:
                points.append(0)
                colors.append(" ")

        # Split the board into top and bottom halves
        bottom_board = points[:12][::-1]
        top_board = points[12:]

        bottom_checkers_color = colors[:12][::-1]
        top_checkers_color = colors[12:]

        assert len(bottom_board) + len(top_board) == 24
        assert len(bottom_checkers_color) + len(top_checkers_color) == 24

        # Print the board headers
        print(
            "| 12 | 13 | 14 | 15 | 16 | 17 | BAR | 18 | 19 | 20 | 21 | 22 | 23 | OFF |"
        )
        print(
            f"|------------Outer Board-------------|     |-----------P={TOKEN.get(Player.PLAYER2, '?')} Home Board----------|     |"
        )

        # Print the top half of the board
        self.print_half_board(
            top_board, top_checkers_color, Player.PLAYER2, reversed_=True
        )
        print(
            "|------------------------------------|     |-----------------------------------|     |"
        )

        # Print the bottom half of the board
        self.print_half_board(
            bottom_board, bottom_checkers_color, Player.PLAYER1, reversed_=False
        )
        print(
            f"|------------Outer Board-------------|     |-----------P={TOKEN.get(Player.PLAYER1, '?')} Home Board----------|     |"
        )
        print(
            "| 11 | 10 | 9  | 8  | 7  | 6  | BAR | 5  | 4  | 3  | 2  | 1  | 0  | OFF |\n"
        )

    def print_half_board(self, half_board, checkers_color, player, reversed_=False):
        player_idx = player.value

        # Determine the maximum number of checkers in this half-board, bar, or borne_off
        max_half = max(half_board) if half_board else 0
        max_bar = self.board.tensor[player_idx, Position.BAR.value].item()
        max_borne_off = self.board.tensor[player_idx, Position.BEAR_OFF.value].item()
        max_length = max(max_half, max_bar, max_borne_off)

        # Start printing rows for the current half of the board
        for i in range(max_length):
            # Determine the level of checkers to display
            row = []
            for count, color in zip(half_board, checkers_color):
                if count > i:
                    row.append(color)
                else:
                    row.append(" ")

            # Bar and Off sections
            bar = (
                f"{TOKEN[player]}"
                if self.board.tensor[player_idx, Position.BAR.value].item() > i
                else " "
            )
            off = (
                f"{TOKEN[player]}"
                if self.board.tensor[player_idx, Position.BEAR_OFF.value].item() > i
                else " "
            )

            # Construct the full row with properly aligned columns
            row_display = (
                " | ".join(f"{r:^3}" for r in row[:6])
                + f" | {bar:^3} | "
                + " | ".join(f"{r:^3}" for r in row[6:])
                + f" | {off:^3} |"
            )
            print(f"|  {row_display}")

    def seed(self, seed=None):
        """
        Sets the seed for reproducibility.
        """
        torch.manual_seed(seed)
        if seed is not None:
            np.random.seed(seed)

    def check_for_gammon(self, player: Player) -> bool:
        """
        Checks if the opponent has not borne off any checkers,
        which qualifies as a gammon.
        """
        opponent = get_opponent(player)
        # Correctly access Channel 3 (Borne-off) for the opponent
        opponent_borne_off = self.board.tensor[3, opponent.value].item()
        return opponent_borne_off == 0

    def check_for_backgammon(self, player: Player) -> bool:
        """
        Checks if the opponent has not borne off any checkers and
        has checkers either in the player's home board or on the bar,
        which qualifies as a backgammon.
        """
        opponent = get_opponent(player)
        opponent_idx = opponent.value

        # Correctly access Channel 3 (Borne-off) for the opponent
        opponent_borne_off = self.board.tensor[3, opponent_idx].item()
        if opponent_borne_off > 0:
            return False  # Opponent has borne off at least one checker

        # Define the home board range for the current player
        if player == Player.PLAYER1:
            home_board_indices = range(18, 24)  # Points 18-23
        else:
            home_board_indices = range(0, 6)  # Points 0-5

        # Check if opponent has any checkers in the player's home board
        for idx in home_board_indices:
            if self.board.tensor[opponent.value, idx].item() > 0:
                return True  # Opponent has a checker in player's home board

        # Correctly access Channel 2 (Bar) for the opponent
        opponent_bar = self.board.tensor[2, opponent_idx].item()
        if opponent_bar > 0:
            return True  # Opponent has checkers on the bar

        return False  # No checkers in home board or on the bar
