from typing import List, Set
from src.board.board import Board
from src.moves.move_types import FullMove
from src.players.player import Player
from src.utils.serialization import execute_move_on_board_copy, add_unique_board
from src.moves.move_logic import get_moves_with_one_die


def handle_non_doubles(
    board: Board,
    roll: List[int],
    full_moves: List[FullMove],
    unique_boards: Set[int],
    player: Player,
    reverse: bool = False,
):
    dice_order = [roll[1], roll[0]] if reverse else [roll[0], roll[1]]

    first_die_moves = get_moves_with_one_die(
        board=board, die_value=dice_order[0], player=player
    )
    all_second_die_moves_empty = True

    # Check if all second die moves are empty
    for initial_move in first_die_moves:
        resulting_board = execute_move_on_board_copy(
            board=board, sub_move=initial_move, player=player
        )
        second_die_moves = get_moves_with_one_die(
            board=resulting_board, die_value=dice_order[1], player=player
        )

        if second_die_moves:
            all_second_die_moves_empty = False
            break

    for initial_move in first_die_moves:
        # Apply the initial move and get the resulting board state
        resulting_board = execute_move_on_board_copy(
            board=board, sub_move=initial_move, player=player
        )
        # Generate all possible moves for the second die roll based on the resulting board
        second_die_moves = get_moves_with_one_die(
            board=resulting_board, die_value=dice_order[1], player=player
        )

        if all_second_die_moves_empty:
            add_unique_board(
                resulting_board, [initial_move], full_moves, unique_boards, player
            )
        else:
            for follow_up_move in second_die_moves:
                board_after_second_move = execute_move_on_board_copy(
                    board=resulting_board, sub_move=follow_up_move, player=player
                )
                add_unique_board(
                    board_after_second_move,
                    [initial_move, follow_up_move],
                    full_moves,
                    unique_boards,
                    player,
                )


def handle_doubles(
    board: Board,
    die_value: int,
    full_moves: List[FullMove],
    unique_boards: Set[int],
    player: Player,
):
    single_die_moves = get_moves_with_one_die(board, die_value, player)
    for first_move in single_die_moves:
        first_board = execute_move_on_board_copy(board, first_move, player)
        second_die_moves = get_moves_with_one_die(first_board, die_value, player)
        if not second_die_moves and len(single_die_moves) == 1:
            add_unique_board(
                first_board, [first_move], full_moves, unique_boards, player
            )
        for second_move in second_die_moves:
            second_board = execute_move_on_board_copy(first_board, second_move, player)
            third_die_moves = get_moves_with_one_die(second_board, die_value, player)
            if not third_die_moves and len(second_die_moves) == 1:
                add_unique_board(
                    second_board,
                    [first_move, second_move],
                    full_moves,
                    unique_boards,
                    player,
                )
            for third_move in third_die_moves:
                third_board = execute_move_on_board_copy(
                    second_board, third_move, player
                )
                fourth_die_moves = get_moves_with_one_die(
                    third_board, die_value, player
                )
                if not fourth_die_moves and len(third_die_moves) == 1:
                    add_unique_board(
                        third_board,
                        [first_move, second_move, third_move],
                        full_moves,
                        unique_boards,
                        player,
                    )
                for fourth_move in fourth_die_moves:
                    final_board = execute_move_on_board_copy(
                        third_board, fourth_move, player
                    )
                    add_unique_board(
                        final_board,
                        [first_move, second_move, third_move, fourth_move],
                        full_moves,
                        unique_boards,
                        player,
                    )
