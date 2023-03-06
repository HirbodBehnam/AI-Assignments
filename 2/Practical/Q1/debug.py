from Board import BoardUtility
from alpha_beta import minimax
import numpy as np
from Game import XO
from Player import HumanPlayer, RandomPlayer, MiniMaxPlayer, MiniMaxProbPlayer

INT_INF = 2147483647

#board = np.array([[1,1,2],[0,0,0],[0,0,2]], dtype=np.int8)
#_, move = minimax(board, 1, 1, 4, True, -INT_INF, INT_INF)
#print("For", board, "choose", move)
#exit(0)


player1 = MiniMaxPlayer(1, depth=4)
#player2 = RandomPlayer(2)
player2 = MiniMaxProbPlayer(2, depth=3, prob_stochastic=0.8)
wins = 0
lost = 0
draw = 0
for _ in range(50):
    board = np.zeros((3, 3))
    turn = 0 # np.random.randint(0, 2)
    print('player1 is O. player2 is X.')
    print(f'player{turn + 1} goes first.')
    while True:
        if turn == 0:
            row, col = player1.play(board)
            BoardUtility.make_move(board, row, col, player1.piece)
        elif turn == 1:
            row, col = player2.play(board)
            BoardUtility.make_move(board, row, col, player2.piece)
        turn = 1 - turn
        if BoardUtility.has_player_won(board, 1):
            print("PLAYER 1 WINS!")
            wins += 1
            break
        if BoardUtility.has_player_won(board, 2):
            print("PLAYER 2 WINS!")
            lost += 1
            break
        if BoardUtility.is_draw(board):
            draw += 1
            print("NO ONE WON DRAW!")
            break

print(f'minimax player vs random player win={wins}, lose={lost}, draw={draw}')