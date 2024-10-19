# # backgammon/utils/tests.py

# import pickle
# from dataclasses import dataclass, field
# from typing import List
# from src.board.board_class import Board
# from src.players.player import Player
# from src.board.owned_state import OwnedState
# from src.utils.serialization import process_board
# import copy
# import concurrent.futures


# def test_comprehensive():
#     print("Starting comprehensive pickle and ProcessPoolExecutor tests...")

#     # Local pickle test
#     print("Testing local pickling and equality...")
#     original_board = Board()
#     serialized_board = pickle.dumps(original_board)
#     deserialized_board = pickle.loads(serialized_board)
#     assert (
#         original_board == deserialized_board
#     ), "Board instances do not match after pickling!"
#     print("Local pickling test passed.")

#     # Prepare multiple boards with varying states
#     boards = []
#     for i in range(10):
#         new_board = copy.copy(original_board)
#         # Modify some state for diversity
#         new_board.borne_off[Player.PLAYER1] += i
#         new_board.bar[Player.PLAYER2] += i % 3
#         boards.append(new_board)

#     # Process boards in parallel
#     print("Testing ProcessPoolExecutor with multiple Board instances...")
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [executor.submit(process_board, b, Player.PLAYER1) for b in boards]
#         results = [f.result() for f in concurrent.futures.as_completed(futures)]

#     # Verify results
#     expected_results = [b.borne_off[Player.PLAYER1] for b in boards]
#     # Since as_completed does not guarantee order, sort both lists before comparison
#     assert sorted(results) == sorted(
#         expected_results
#     ), "ProcessPoolExecutor results mismatch!"
#     print("ProcessPoolExecutor test passed.")

#     print("All comprehensive tests passed successfully.")


# def test_owned_state_pickle():
#     print("Testing OwnedState pickling...")
#     original = OwnedState(count=3, player=Player.PLAYER1)
#     serialized = pickle.dumps(original)
#     deserialized = pickle.loads(serialized)
#     assert original == deserialized, "OwnedState instances do not match after pickling!"
#     print("OwnedState pickling test passed.")


# def test_board_pickle():
#     print("Testing Board pickling...")
#     original_board = Board()
#     serialized = pickle.dumps(original_board)
#     deserialized_board = pickle.loads(serialized)
#     assert (
#         original_board == deserialized_board
#     ), "Board instances do not match after pickling!"
#     print("Board pickling test passed.")
