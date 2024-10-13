from dataclasses import dataclass
from enum import Enum, auto
from typing import List
from src.board.board_class import Board
from src.players.player import Player
from src.moves.get_all_moves import get_all_possible_moves


# Mapping from Player to tensor index
PLAYER_TO_INDEX = {
    Player.PLAYER1: 0,
    Player.PLAYER2: 1,
}


# Assuming other necessary functions and classes are defined here...
# For brevity, they are not included in this snippet.


def print_board(board: Board):
    """
    Prints the current state of the board in a readable format.
    """
    print("Board State:")
    for player in [Player.PLAYER1, Player.PLAYER2]:
        player_idx = PLAYER_TO_INDEX[player]
        checkers = board.points[player_idx].tolist()
        bar = board.bar[player_idx].item()
        borne_off = board.borne_off[player_idx].item()
        print(f"{player.name}:")
        print("  Points:")
        for idx, count in enumerate(checkers):
            print(f"    Point {idx}: {count}")
        print(f"  Bar: {bar}")
        print(f"  Borne Off: {borne_off}")
    print("-" * 40)


def test_board_class():
    """
    Tests the Board class by adding and removing checkers and printing the board state.
    """
    print("=== Testing Board Class ===")

    # Initialize the board
    board = Board()
    print("Initial Board:")
    print_board(board)

    # Define players
    player1 = Player.PLAYER1
    player2 = Player.PLAYER2

    # Add a checker for Player1 at point 5
    print(f"Adding a checker for {player1.name} at point 5.")
    board.add_checker(index=5, player=player1)
    print_board(board)

    # Remove a checker for Player2 from point 7
    print(f"Removing a checker for {player2.name} from point 7.")
    board.remove_checker(index=7, player=player2)
    print_board(board)

    # Attempt to remove a checker from an empty point (should trigger a warning)
    print(f"Attempting to remove a checker for {player2.name} from empty point 7.")
    board.remove_checker(index=7, player=player2)
    print_board(board)

    print("=== Board Class Test Completed ===\n")


def test_get_all_possible_moves():
    """
    Tests the get_all_possible_moves function by setting up a board state,
    performing a dice roll, and printing all available moves.
    """
    print("=== Testing get_all_possible_moves ===")

    # Initialize the board
    board = Board()
    print("Initial Board:")
    print_board(board)

    # Define players
    player1 = Player.PLAYER1
    player2 = Player.PLAYER2

    # Set up a specific board state for testing
    # For example, Player1 has checkers at points 0, 5, and 11
    board.points[PLAYER_TO_INDEX[player1], 0] = 2
    board.points[PLAYER_TO_INDEX[player1], 5] = 3
    board.points[PLAYER_TO_INDEX[player1], 11] = 5

    # Player2 has checkers at points 7, 12, and 23
    board.points[PLAYER_TO_INDEX[player2], 7] = 3
    board.points[PLAYER_TO_INDEX[player2], 12] = 5
    board.points[PLAYER_TO_INDEX[player2], 23] = 2

    print("Custom Board Setup:")
    print_board(board)

    # Define a dice roll
    roll_result = [3, 5]
    print("You Rolled:", roll_result)

    # Define the current player
    current_player = player1

    # Generate all possible moves
    legal_moves = get_all_possible_moves(
        player=current_player, board_copy=board, roll_result=roll_result
    )

    # Print available moves
    print("\nAvailable moves:")
    for i, move in enumerate(legal_moves):
        # Generate the description for each SubMove
        moves_description = ", ".join(
            f"[{'bar' if sub_move.start_index == -1 else sub_move.start_index}, "
            f"{'off' if sub_move.end_index == -2 else sub_move.end_index}, "
            f"{'*' if sub_move.hits_blot else '-'}]"
            for sub_move in move.sub_move_commands
        )

        # Format the full move string
        full_move_str = f"Full Move ({move.player.name}): {moves_description}"

        # Print the formatted move
        print(f"Move {i}: {full_move_str}")

    print("=== get_all_possible_moves Test Completed ===\n")


# Execute the tests
if __name__ == "__main__":
    test_board_class()
    test_get_all_possible_moves()
