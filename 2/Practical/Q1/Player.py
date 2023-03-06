from Board import BoardUtility
from alpha_beta import INT_INF, minimax, minimax_prob
import random
import numpy as np
import numpy.typing as npt

class Player:
    def __init__(self, player_piece):
        self.piece = player_piece

    def play(self, board):
        return 0


class RandomPlayer(Player):
    def play(self, board):
        return random.choice(BoardUtility.get_valid_locations(board))


class HumanPlayer(Player):
    def play(self, board):
        move = int(input("input the next column index 1 to 9:")) - 1
        row = move // 3
        col = move % 3
        return row, col


class MiniMaxPlayer(Player):
    def __init__(self, player_piece, depth=5):
        super().__init__(player_piece)
        self.depth = depth

    def play(self, board: npt.NDArray[np.float64]) -> tuple[int, int]:
        """
        Inputs : 
           board : 3*3 numpy array. 0 for empty cell, 1 and 2 for cells contains a piece.
        return the next move (i,j) of the player based on minimax algorithm.
        """
        betterBoard = board.astype(np.int8) # fuck floating point numbers
        _, move = minimax(betterBoard, self.piece, self.piece, self.depth, True, -INT_INF, INT_INF)
        print("For", betterBoard, "choose", move)
        return move


class MiniMaxProbPlayer(Player):
    def __init__(self, player_piece, depth=5, prob_stochastic=0.1):
        super().__init__(player_piece)
        self.depth = depth
        self.prob_stochastic = prob_stochastic

    def play(self, board):
        """
        Inputs : 
           board : 3*3 numpy array. 0 for empty cell, 1 and 2 for cells contains a piece.
        same as above but each time you are playing as max choose a random move instead of the best move
        with probability self.prob_stochastic.
        """
        betterBoard = board.astype(np.int8) # fuck floating point numbers
        _, move = minimax_prob(betterBoard, self.piece, self.piece, self.depth, True, -INT_INF, INT_INF, self.prob_stochastic)
        print("For", betterBoard, "choose (rand)", move)
        return move
