import gym
import numpy as np
import torch
from gym import spaces
from src.board.board_class import Board
from src.players.player import Player
from src.moves.handle_moves import execute_move_on_board_copy
from src.moves.get_all_moves import get_all_possible_moves
from src.ai.batching import (
    apply_moves_and_get_features_in_batch,
)

PLAYER_TO_INDEX = {
    Player.PLAYER1: 0,
    Player.PLAYER2: 1,
}

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

# Token Representation for Players
TOKEN = {
    Player.PLAYER1: "H",  # Representing PLAYER1 with "H"
    Player.PLAYER2: "A",  # Representing PLAYER2 with "A"
}


class BackgammonEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, match_length=15, max_legal_moves=500, device=None):
        super(BackgammonEnv, self).__init__()

        self.match_length = match_length
        self.player_scores = {Player.PLAYER1: 0, Player.PLAYER2: 0}
        self.current_match_winner = None

        # Set the device
        self.device = device if device is not None else torch.device("cpu")

        self.board = Board(device=self.device)
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
        self.board = Board(device=self.device)
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
        if self.game_over:
            observation = self.reset()
            return observation, torch.tensor(0.0, device=self.device), True, {}

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
            return observation, reward, done, {"info": "No legal actions, turn passed"}

        # Validate action
        if not self.action_mask[action]:
            # Invalid action selected
            reward = torch.tensor(REWARD_INVALID_ACTION, device=self.device)
            done = False
            print(f"Invalid action selected: {action}. Assigned reward: {reward}")
            observation = self.get_observation()
            return observation, reward, done, {"info": "Invalid action"}

        # Execute the Selected Move by applying the corresponding FullMove
        selected_move = self.legal_moves[action]
        self.board = execute_move_on_board_copy(
            self.board, selected_move, self.current_player
        )

        # Check for game over
        if self.board.borne_off[PLAYER_TO_INDEX[self.current_player]] == 15:
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
            info = {"winner": self.current_player, "game_score": game_score}
            self.player_scores[self.current_player] += game_score
            self.game_over = True
            done = True

            # Check if match is over
            if self.player_scores[self.current_player] >= self.match_length:
                # Match is over
                self.current_match_winner = self.current_player
                self.match_over = True
        else:
            info = {}
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
            board_copy=self.board,
            roll_result=self.roll_result,
        )

        # Generate legal board features for the action mask
        if self.legal_moves:
            self.legal_board_features = apply_moves_and_get_features_in_batch(
                self.board, self.legal_moves, self.current_player, device=self.device
            )
        else:
            self.legal_board_features = torch.empty(
                (0, 198), dtype=torch.float32, device=self.device
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
            player1_checkers = self.board.points[
                PLAYER_TO_INDEX[Player.PLAYER1], point_idx
            ].item()
            player2_checkers = self.board.points[
                PLAYER_TO_INDEX[Player.PLAYER2], point_idx
            ].item()

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
        player_idx = PLAYER_TO_INDEX[player]

        # Determine the maximum number of checkers in this half-board, bar, or borne_off
        max_half = max(half_board) if half_board else 0
        max_bar = self.board.bar[player_idx].item()
        max_borne_off = self.board.borne_off[player_idx].item()
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
            bar = f"{TOKEN[player]}" if self.board.bar[player_idx].item() > i else " "
            off = (
                f"{TOKEN[player]}"
                if self.board.borne_off[player_idx].item() > i
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
        opponent = Player.PLAYER2 if player == Player.PLAYER1 else Player.PLAYER1
        opponent_borne_off = self.board.borne_off[PLAYER_TO_INDEX[opponent]].item()
        return opponent_borne_off == 0

    def check_for_backgammon(self, player: Player) -> bool:
        """
        Checks if the opponent has not borne off any checkers and
        has checkers either in the player's home board or on the bar,
        which qualifies as a backgammon.
        """
        opponent = Player.PLAYER2 if player == Player.PLAYER1 else Player.PLAYER1
        opponent_idx = PLAYER_TO_INDEX[opponent]

        # Check if opponent has borne off any checkers
        opponent_borne_off = self.board.borne_off[opponent_idx].item()
        if opponent_borne_off > 0:
            return False  # Opponent has borne off at least one checker

        # Define the home board range for the current player
        if player == Player.PLAYER1:
            home_board_indices = range(18, 24)  # Points 18-23
        else:
            home_board_indices = range(0, 6)  # Points 0-5

        # Check if opponent has any checkers in the player's home board
        for idx in home_board_indices:
            if self.board.points[opponent_idx, idx].item() > 0:
                return True  # Opponent has a checker in player's home board

        # Check if opponent has any checkers on the bar
        opponent_bar = self.board.bar[opponent_idx].item()
        if opponent_bar > 0:
            return True  # Opponent has checkers on the bar

        return False  # No checkers in home board or on the bar
