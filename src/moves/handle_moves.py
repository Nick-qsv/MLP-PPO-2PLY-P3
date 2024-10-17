from typing import List, Set
from src.board.board_class import Board
from src.moves.move_types import FullMove, SubMove
from src.players.player import Player
from src.moves.move_logic import get_moves_with_one_die
from src.constants import PLAYER_TO_INDEX, BEAR_OFF_INDEX


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
        resulting_board = execute_sub_move_on_board(
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
        resulting_board = execute_sub_move_on_board(
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
                board_after_second_move = execute_sub_move_on_board(
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
        first_board = execute_sub_move_on_board(board, first_move, player)
        second_die_moves = get_moves_with_one_die(first_board, die_value, player)
        if not second_die_moves and len(single_die_moves) == 1:
            add_unique_board(
                first_board, [first_move], full_moves, unique_boards, player
            )
        for second_move in second_die_moves:
            second_board = execute_sub_move_on_board(first_board, second_move, player)
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
                third_board = execute_sub_move_on_board(
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
                    final_board = execute_sub_move_on_board(
                        third_board, fourth_move, player
                    )
                    add_unique_board(
                        final_board,
                        [first_move, second_move, third_move, fourth_move],
                        full_moves,
                        unique_boards,
                        player,
                    )


def execute_full_move_on_board_copy(
    board: Board, full_move: FullMove, player: Player
) -> Board:
    """
    Executes a full move (composed of sub-moves) on a copy of the board.
    """
    board_copy = board.copy()
    for sub_move in full_move.sub_move_commands:
        execute_sub_move_on_board(board_copy, sub_move, player)
    return board_copy


def execute_sub_move_on_board(board: Board, sub_move: SubMove, player: Player) -> Board:
    """
    Executes a sub-move on the board.
    """
    player_idx = PLAYER_TO_INDEX[player]
    opponent_idx = 1 - player_idx

    # Remove the checker from the start index
    board.remove_checker(sub_move.start_index, player)

    if sub_move.hits_blot:
        # Remove opponent's checker from the end index and place it on the bar
        board.points[opponent_idx, sub_move.end_index] -= 1
        board.bar[opponent_idx] += 1

    # Add the checker to the end index or bear off
    if sub_move.end_index == BEAR_OFF_INDEX:
        # Bear off
        board.borne_off[player_idx] += 1
    else:
        board.add_checker(sub_move.end_index, player)
    return board


def board_hash(board: Board) -> int:
    """
    Generates a hashable representation of the board.
    """
    return hash(
        (
            tuple(board.points.view(-1).tolist()),
            tuple(board.bar.tolist()),
            tuple(board.borne_off.tolist()),
        )
    )


def add_unique_board(
    board: Board,
    moves: List[SubMove],
    full_moves: List[FullMove],
    unique_boards: Set[int],
    player: Player,
):
    board_h = board_hash(board)
    if board_h not in unique_boards:
        unique_boards.add(board_h)
        full_move = FullMove(sub_move_commands=moves.copy(), player=player)
        full_moves.append(full_move)
