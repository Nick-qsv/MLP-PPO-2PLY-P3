from typing import List, Set
from src.board.immutable_board import (
    ImmutableBoard,
    board_hash,
    execute_sub_move_on_board,
)
from src.moves.move_types import FullMove, SubMove
from src.players.player import Player
from src.moves.move_logic import get_moves_with_one_die
import logging

logging.basicConfig(
    level=logging.WARNING,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def handle_non_doubles(
    board: ImmutableBoard,
    roll: List[int],
    full_moves: List[FullMove],
    unique_boards: Set[int],
    player: Player,
    reverse: bool = False,
):
    """
    Handles move generation for non-double die rolls.

    This function generates all possible move sequences for a player based on two distinct die values.
    It first determines the order of die application (reversed if specified), generates initial moves
    using the first die, and then generates subsequent moves using the second die based on the resulting
    board states. Unique full move sequences are added to the full_moves list.

    Args:
        board (ImmutableBoard): The current state of the board.
        roll (List[int]): A list containing two die values.
        full_moves (List[FullMove]): A list to store all unique full move sequences.
        unique_boards (Set[int]): A set to keep track of already processed board states.
        player (Player): The player for whom to generate moves.
        reverse (bool, optional): If True, reverses the order of die application. Defaults to False.
    """
    # Determine the order of dice based on the reverse flag
    dice_order = [roll[1], roll[0]] if reverse else [roll[0], roll[1]]

    # Generate all possible initial moves using the first die
    first_die_moves = get_moves_with_one_die(
        board=board, die_value=dice_order[0], player=player
    )

    # Flag to check if all second die moves are empty
    all_second_die_moves_empty = True

    # Check if any second die moves are possible after applying initial moves
    for initial_move in first_die_moves:
        resulting_board = execute_sub_move_on_board(
            board=board, sub_move=initial_move, player=player
        )
        second_die_moves = get_moves_with_one_die(
            board=resulting_board, die_value=dice_order[1], player=player
        )

        if second_die_moves:
            all_second_die_moves_empty = False
            break  # At least one second die move is possible

    # Iterate through all initial moves to generate full move sequences
    for initial_move in first_die_moves:
        # Apply the initial move to get the resulting board state
        resulting_board = execute_sub_move_on_board(
            board=board, sub_move=initial_move, player=player
        )

        # Generate all possible second moves based on the second die
        second_die_moves = get_moves_with_one_die(
            board=resulting_board, die_value=dice_order[1], player=player
        )

        if all_second_die_moves_empty:
            # If no second moves are possible, record the initial move as a full move
            add_unique_board(
                board=resulting_board,
                moves=[initial_move],
                full_moves=full_moves,
                unique_boards=unique_boards,
                player=player,
            )
        else:
            # Otherwise, iterate through all possible second moves
            for follow_up_move in second_die_moves:
                # Apply the follow-up move to get the new board state
                board_after_second_move = execute_sub_move_on_board(
                    board=resulting_board, sub_move=follow_up_move, player=player
                )

                # Record the full move sequence (initial move + follow-up move)
                add_unique_board(
                    board=board_after_second_move,
                    moves=[initial_move, follow_up_move],
                    full_moves=full_moves,
                    unique_boards=unique_boards,
                    player=player,
                )


def handle_doubles(
    board: ImmutableBoard,
    die_value: int,
    full_moves: List[FullMove],
    unique_boards: Set[int],
    player: Player,
):
    """
    Handles move generation for double die rolls.

    This function generates all possible move sequences for a player based on a double die roll.
    Since doubles allow the player to make four moves instead of two, the function recursively
    generates move sequences up to four sub-moves, ensuring that each sequence is unique.

    Args:
        board (ImmutableBoard): The current state of the board.
        die_value (int): The value of the die rolled (both die values are the same).
        full_moves (List[FullMove]): A list to store all unique full move sequences.
        unique_boards (Set[int]): A set to keep track of already processed board states.
        player (Player): The player for whom to generate moves.
    """
    # Generate all possible first moves using the die value
    single_die_moves = get_moves_with_one_die(board, die_value, player)

    for first_move in single_die_moves:
        # Apply the first move to get the resulting board state
        first_board = execute_sub_move_on_board(board, first_move, player)

        # Generate all possible second moves using the same die value
        second_die_moves = get_moves_with_one_die(first_board, die_value, player)

        if not second_die_moves and len(single_die_moves) == 1:
            # If no second moves are possible and only one first move exists, record it
            add_unique_board(
                board=first_board,
                moves=[first_move],
                full_moves=full_moves,
                unique_boards=unique_boards,
                player=player,
            )

        for second_move in second_die_moves:
            # Apply the second move to get the new board state
            second_board = execute_sub_move_on_board(first_board, second_move, player)

            # Generate all possible third moves using the same die value
            third_die_moves = get_moves_with_one_die(second_board, die_value, player)

            if not third_die_moves and len(second_die_moves) == 1:
                # If no third moves are possible and only one second move exists, record the sequence
                add_unique_board(
                    board=second_board,
                    moves=[first_move, second_move],
                    full_moves=full_moves,
                    unique_boards=unique_boards,
                    player=player,
                )

            for third_move in third_die_moves:
                # Apply the third move to get the new board state
                third_board = execute_sub_move_on_board(
                    second_board, third_move, player
                )

                # Generate all possible fourth moves using the same die value
                fourth_die_moves = get_moves_with_one_die(
                    third_board, die_value, player
                )

                if not fourth_die_moves and len(third_die_moves) == 1:
                    # If no fourth moves are possible and only one third move exists, record the sequence
                    add_unique_board(
                        board=third_board,
                        moves=[first_move, second_move, third_move],
                        full_moves=full_moves,
                        unique_boards=unique_boards,
                        player=player,
                    )

                for fourth_move in fourth_die_moves:
                    # Apply the fourth move to get the final board state
                    final_board = execute_sub_move_on_board(
                        third_board, fourth_move, player
                    )

                    # Record the full move sequence (four sub-moves)
                    add_unique_board(
                        board=final_board,
                        moves=[first_move, second_move, third_move, fourth_move],
                        full_moves=full_moves,
                        unique_boards=unique_boards,
                        player=player,
                    )


def add_unique_board(
    board: ImmutableBoard,
    moves: List[SubMove],
    full_moves: List[FullMove],
    unique_boards: Set[int],
    player: Player,
):
    """
    Adds a unique full move sequence to the list of full moves.

    This function checks if the resulting board state has already been encountered by
    verifying its hash against the unique_boards set. If the board state is unique,
    it adds the board's hash to the set and appends the move sequence to full_moves.

    Args:
        board (ImmutableBoard): The resulting state of the board after applying moves.
        moves (List[SubMove]): The list of sub-moves that constitute the full move.
        full_moves (List[FullMove]): The list to store all unique full move sequences.
        unique_boards (Set[int]): A set to keep track of already processed board states.
        player (Player): The player who made the moves.
    """
    board_h = board_hash(board)
    if board_h not in unique_boards:
        unique_boards.add(board_h)
        # Create a new FullMove instance with a copy of the current move sequence
        full_move = FullMove(sub_move_commands=moves.copy(), player=player)
        full_moves.append(full_move)
        # Optional: Log the addition for debugging purposes
        logger.debug(f"Added unique full move: {full_move}")
