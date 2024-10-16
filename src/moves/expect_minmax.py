import copy
from typing import List, Tuple, Dict
import torch  # pylint: disable=import-error
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from src.board.board_class import Board
from src.players.player import Player
from src.moves.move_types import SubMove, FullMove
from src.moves.get_all_moves import get_all_possible_moves
from src.moves.handle_moves import execute_move_on_board_copy
from src.moves.get_all_dice_rolls import get_all_dice_rolls_tensor

# from src.utils.dice import get_all_dice_rolls

# generate_all_board_features_non_batch, get_all_dice_rolls, execute_move_on_board_copy


# Placeholder evaluation function
def evaluate_board(board: Board, current_player: Player) -> float:
    """
    Placeholder evaluation function.
    Replace this with a proper evaluation metric.
    """
    # Example: difference in borne_off checkers
    return (
        board.borne_off[current_player]
        - board.borne_off[
            Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1
        ]
    )


def expectiminimax_alpha_beta(
    board: Board,
    current_player: Player,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
) -> float:
    """
    Expectiminimax algorithm with alpha-beta pruning for Backgammon.

    Parameters:
    - board (Board): The current board state.
    - current_player (Player): The player to evaluate for.
    - depth (int): Current depth in the search tree.
    - alpha (float): Alpha value for pruning.
    - beta (float): Beta value for pruning.
    - maximizing_player (bool): True if the current node is a maximizing node.

    Returns:
    - float: The evaluation score of the board.
    """
    if depth == 0:
        return evaluate_board(board, current_player)

    if maximizing_player:
        max_eval = float("-inf")
        # Generate all possible dice rolls
        dice_rolls, dice_probs = get_all_dice_rolls_tensor()
        for dice, prob in zip(dice_rolls, dice_probs):
            # Generate all possible moves for current_player given the dice roll
            possible_moves = get_all_possible_moves(
                player=current_player, board_copy=copy.copy(board), roll_result=dice
            )
            if not possible_moves:
                # No possible moves, possibly a pass
                eval = evaluate_board(board, current_player)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
                continue

            for move in possible_moves:
                # Apply the move to get the new board state
                new_board = copy.deepcopy(board)
                for sub_move in move.sub_move_commands:
                    new_board = execute_move_on_board_copy(
                        board=new_board, sub_move=sub_move, player=current_player
                    )
                # Recurse with reduced depth
                eval = expectiminimax_alpha_beta(
                    board=new_board,
                    current_player=current_player,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing_player=False,  # Next level is minimizing
                )
                # Since this is a chance node, multiply by probability
                max_eval = max(max_eval, eval * prob)
                alpha = max(alpha, eval * prob)
                if beta <= alpha:
                    break  # Beta cut-off
        return max_eval

    else:
        # Minimizing player: opponent
        min_eval = float("inf")
        opponent = (
            Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1
        )
        # Generate all possible dice rolls for opponent
        dice_rolls, dice_probs = get_all_dice_rolls_tensor()
        for dice, prob in zip(dice_rolls, dice_probs):
            # Generate all possible moves for opponent given the dice roll
            possible_moves = get_all_possible_moves(
                player=opponent, board_copy=copy.copy(board), roll_result=dice
            )
            if not possible_moves:
                # No possible moves, possibly a pass
                eval = evaluate_board(board, current_player)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
                continue

            for move in possible_moves:
                # Apply the move to get the new board state
                new_board = copy.deepcopy(board)
                for sub_move in move.sub_move_commands:
                    new_board = execute_move_on_board_copy(
                        board=new_board, sub_move=sub_move, player=opponent
                    )
                # Recurse with reduced depth
                eval = expectiminimax_alpha_beta(
                    board=new_board,
                    current_player=current_player,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing_player=True,  # Next level is maximizing
                )
                # Since this is a chance node, multiply by probability
                min_eval = min(min_eval, eval * prob)
                beta = min(beta, eval * prob)
                if beta <= alpha:
                    break  # Alpha cut-off
        return min_eval


def choose_best_move(board: Board, current_player: Player) -> FullMove:
    """
    Chooses the best move for the current player using Expectiminimax with alpha-beta pruning.

    Parameters:
    - board (Board): The current board state.
    - current_player (Player): The player to choose a move for.

    Returns:
    - FullMove: The best move found.
    """
    best_score = float("-inf")
    best_move = None
    alpha = float("-inf")
    beta = float("inf")

    # Assume dice roll is already determined; if not, integrate dice rolls here
    # For simplicity, we'll iterate over all possible dice rolls and average the results
    dice_rolls, dice_probs = get_all_dice_rolls_tensor()

    for dice, prob in zip(dice_rolls, dice_probs):
        possible_moves = get_all_possible_moves(
            player=current_player, board_copy=copy.copy(board), roll_result=dice
        )
        if not possible_moves:
            continue  # No moves for this dice roll

        for move in possible_moves:
            # Apply the move to get the new board state
            new_board = copy.deepcopy(board)
            for sub_move in move.sub_move_commands:
                new_board = execute_move_on_board_copy(
                    board=new_board, sub_move=sub_move, player=current_player
                )
            # Evaluate the move using Expectiminimax
            move_score = expectiminimax_alpha_beta(
                board=new_board,
                current_player=current_player,
                depth=1,  # 2-ply: current move + opponent's response
                alpha=alpha,
                beta=beta,
                maximizing_player=False,
            )
            # Since dice rolls have probabilities, weigh the score
            weighted_score = move_score * prob
            if weighted_score > best_score:
                best_score = weighted_score
                best_move = move
            alpha = max(alpha, weighted_score)
            if beta <= alpha:
                break  # Beta cut-off

    return best_move


# Example usage:
if __name__ == "__main__":
    initial_board = Board()
    current_player = Player.PLAYER1
    best_move = choose_best_move(initial_board, current_player)
    print(f"Best Move: {best_move}")
