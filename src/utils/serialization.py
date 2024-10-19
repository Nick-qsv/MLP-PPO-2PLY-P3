# # backgammon/utils/serialization.py

# import pickle
# from typing import List, Set
# from src.board.board_class import Board
# from src.players.player import Player

# # todo: add process pool to execute move on board copy.  they do not have to be unique (they already are unique)


# def process_board(board: Board, player: Player):
#     """
#     For testing, perform serialization and return the number of checkers borne off by the player.

#     Parameters:
#     - board (Board): The board to process.
#     - player (Player): The player whose borne off checkers to count.

#     Returns:
#     - int: Number of checkers borne off by the player.
#     """
#     serialized = pickle.dumps(board)
#     deserialized = pickle.loads(serialized)
#     return deserialized.borne_off[player]
