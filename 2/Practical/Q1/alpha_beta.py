from Board import BoardUtility
import random
import numpy as np
import numpy.typing as npt
from typing import Optional

INT_INF = 2147483647
                
def other_player_piece(player_piece: int) -> int:
    if player_piece == 1: 
        return 2
    elif player_piece == 2:
        return 1
    else:
        raise Exception("player piece is " + str(player_piece))

def minimax(board: npt.NDArray[np.int8], starting_player: int, player_piece: int, depth: int, is_maximizing_player: bool, alpha: int, beta: int) -> tuple[int, Optional[tuple[int, int]]]:
    """
    This function run alpha beta puring algorithm and return best next move
    :param board: game board
    :param starting_player: who is maximizing?
    :param player_piece: number of player. 1 or 2
    :param depth: depth of tree
    :param is_maximizing_player: True if you are in a max node otherwise False
    :param alpha: value of alpha
    :param beta: value of beta
    :return best_value, best_move: You have to return best next move and its value
    """
    # Check terminal state
    if BoardUtility.is_terminal_state(board) or depth == 0:
        return (BoardUtility.score_position(board, starting_player), None)
    # Normal stuff just like slides
    allowed_moves = BoardUtility.get_valid_locations(board)
    total_best_move = None
    v = 0
    if is_maximizing_player:
        v = -INT_INF
        for allowed_move in allowed_moves:
            copied_board = np.copy(board)
            copied_board[allowed_move[0], allowed_move[1]] = player_piece
            best_value, _ = minimax(copied_board, starting_player, other_player_piece(player_piece), depth - 1, not is_maximizing_player, alpha, beta)
            if v < best_value:
                v = best_value
                total_best_move = allowed_move
            if v >= beta:
                break
            alpha = max(alpha, v)
    else:
        v = INT_INF
        for allowed_move in allowed_moves:
            copied_board = np.copy(board)
            copied_board[allowed_move[0], allowed_move[1]] = player_piece
            best_value, _ = minimax(copied_board, starting_player, other_player_piece(player_piece), depth - 1, not is_maximizing_player, alpha, beta)
            if v >= best_value:
                v = best_value
                total_best_move = allowed_move
            if v <= alpha:
                break
            beta = min(beta, v)
    # If we have not found anything, return a random place
    if total_best_move == None:
        return (BoardUtility.score_position(board, starting_player), random.choice(BoardUtility.get_valid_locations(board)))
    else: # otherwise we are good
        return (v, total_best_move)



def minimax_prob(board: npt.NDArray[np.int8], starting_player: int, player_piece: int, depth: int, is_maximizing_player: bool, alpha: int, beta: int, prob: float) -> tuple[int, Optional[tuple[int, int]]]:
    """
    This function run alpha beta puring algorithm and return best next move
    :param board: game board
    :param player_piece: number of player. 1 or 2
    :param depth: depth of tree
    :param is_maximizing_player: True if you are in a max node otherwise False
    :param alpha: value of alpha
    :param beta: value of beta
    :param prob: probability of choosing a random action in each max node
    :return best_value, best_move: You have to return best next move and its value
    """
    # Check terminal state
    if BoardUtility.is_terminal_state(board) or depth == 0:
        return (BoardUtility.score_position(board, starting_player), None)
    # Normal stuff just like slides
    allowed_moves = BoardUtility.get_valid_locations(board)
    total_best_move = None
    v = 0
    if is_maximizing_player:
        # Check random move
        if random.uniform(0, 1) < prob:
            return (BoardUtility.score_position(board, starting_player), random.choice(BoardUtility.get_valid_locations(board)))
        v = -INT_INF
        for allowed_move in allowed_moves:
            copied_board = np.copy(board)
            copied_board[allowed_move[0], allowed_move[1]] = player_piece
            best_value, _ = minimax(copied_board, starting_player, other_player_piece(player_piece), depth - 1, not is_maximizing_player, alpha, beta)
            if v < best_value:
                v = best_value
                total_best_move = allowed_move
            if v >= beta:
                break
            alpha = max(alpha, v)
    else:
        v = INT_INF
        for allowed_move in allowed_moves:
            copied_board = np.copy(board)
            copied_board[allowed_move[0], allowed_move[1]] = player_piece
            best_value, _ = minimax(copied_board, starting_player, other_player_piece(player_piece), depth - 1, not is_maximizing_player, alpha, beta)
            if v >= best_value:
                v = best_value
                total_best_move = allowed_move
            if v <= alpha:
                break
            beta = min(beta, v)
    # If we have not found anything, return a random place
    if total_best_move == None:
        return (BoardUtility.score_position(board, starting_player), random.choice(BoardUtility.get_valid_locations(board)))
    else: # otherwise we are good
        return (v, total_best_move)
