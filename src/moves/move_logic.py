# backgammon/moves/move_logic.py

from typing import List
from src.board.board import Board
from src.players.player import Player
from src.moves.move_types import SubMove
from .conditions import (
    valid_move,
    check_if_blot,
    is_valid_entry_at_index,
    check_for_win,
    check_for_bar,
    all_checkers_home,
)
from src.board.board_state import BoardState

PLAYER_TO_INDEX = {
    Player.PLAYER1: 0,
    Player.PLAYER2: 1,
}
# Functions originally from the initial code


def get_moves_with_one_die(board: Board, die_value: int, player: Player) -> list:
    current_board_state = compute_board_state(board, player)

    if current_board_state == BoardState.NORMAL:
        return get_moves_normal(board, die_value, player)
    elif current_board_state == BoardState.ON_BAR:
        return get_moves_bar(board, die_value, player)
    elif current_board_state == BoardState.BEAR_OFF:
        return get_moves_bear_off(board, die_value, player)
    else:
        return []


def get_moves_normal(board: Board, die_value: int, player: Player) -> List[SubMove]:
    moves = []
    direction = 1 if player == Player.PLAYER1 else -1
    number_of_points = 24
    player_idx = PLAYER_TO_INDEX[player]

    for idx in range(number_of_points):
        own_checkers = board.points[player_idx, idx].item()
        if own_checkers > 0:
            destination_idx = idx + die_value * direction
            if valid_move(destination_idx=destination_idx, player=player, board=board):
                sub_move = SubMove(
                    start_index=idx,
                    end_index=destination_idx,
                    hits_blot=check_if_blot(
                        index=destination_idx, player=player, board=board
                    ),
                )
                moves.append(sub_move)
    return moves


def get_moves_bar(board: Board, die_value: int, player: Player) -> List[SubMove]:
    moves = []
    number_of_points = 24
    player_idx = PLAYER_TO_INDEX[player]

    if player == Player.PLAYER1:
        dest_idx = die_value - 1
    else:
        dest_idx = number_of_points - die_value

    if is_valid_entry_at_index(dest_idx, player, board):
        sub_move = SubMove(
            start_index=-1,  # Represents starting from the bar
            end_index=dest_idx,
            hits_blot=check_if_blot(index=dest_idx, player=player, board=board),
        )
        moves.append(sub_move)
    return moves


def get_moves_bear_off(board: Board, die_value: int, player: Player) -> List[SubMove]:
    move_set = []
    number_of_points = 24
    player_idx = PLAYER_TO_INDEX[player]

    if player == Player.PLAYER1:
        home_indexes = list(range(18, 24))
        direction = 1
        last_checker_idx = 18
    else:
        home_indexes = list(range(0, 6))
        direction = -1
        last_checker_idx = 5

    # Normal moves within the home board
    for idx in home_indexes:
        own_checkers = board.points[player_idx, idx].item()
        if own_checkers > 0:
            destination_idx = idx + die_value * direction
            if valid_move(destination_idx=destination_idx, player=player, board=board):
                sub_move = SubMove(
                    start_index=idx,
                    end_index=destination_idx,
                    hits_blot=check_if_blot(
                        index=destination_idx, player=player, board=board
                    ),
                )
                move_set.append(sub_move)

    # Find the index of the farthest checker from the exit point (last checker)
    if player == Player.PLAYER1:
        exit_range = range(18, 24)
        for idx in exit_range:
            if board.points[player_idx, idx].item() > 0:
                last_checker_idx = idx
                break
    else:
        exit_range = range(5, -1, -1)
        for idx in exit_range:
            if board.points[player_idx, idx].item() > 0:
                last_checker_idx = idx
                break

    # Handle bear-off moves
    if player == Player.PLAYER1:
        if last_checker_idx + die_value * direction >= 24:
            sub_move = SubMove(
                start_index=last_checker_idx, end_index=-2, hits_blot=False
            )
            move_set.append(sub_move)

        potential_start_index = 24 - die_value
        if potential_start_index != last_checker_idx:
            if board.points[player_idx, potential_start_index].item() > 0:
                sub_move = SubMove(
                    start_index=potential_start_index, end_index=-2, hits_blot=False
                )
                move_set.append(sub_move)
    else:
        if last_checker_idx + die_value * direction < 0:
            sub_move = SubMove(
                start_index=last_checker_idx, end_index=-2, hits_blot=False
            )
            move_set.append(sub_move)

        potential_start_index = die_value - 1
        if potential_start_index != last_checker_idx:
            if board.points[player_idx, potential_start_index].item() > 0:
                sub_move = SubMove(
                    start_index=potential_start_index, end_index=-2, hits_blot=False
                )
                move_set.append(sub_move)

    return move_set


def compute_board_state(board: Board, player: Player) -> "BoardState":
    if check_for_win(board, player):
        return BoardState.GAME_OVER
    if check_for_bar(board, player):
        return BoardState.ON_BAR
    if all_checkers_home(board, player):
        return BoardState.BEAR_OFF
    return BoardState.NORMAL
