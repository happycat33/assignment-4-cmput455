#!/usr/bin/python3
# /usr/bin/python3
# Set the path to your python3 above

from gtp_connection import GtpConnection
from board_util import GoBoardUtil
from board import GoBoard
import numpy as np

##################### Global Helper Method##############
def play_game(board:GoBoard):
    """
    Run a simulation game to the end fromt the current board
    """
    while True:
        # play a random move for the current player
        color = board.current_player
        move = GoBoardUtil.generate_random_move(board,color)
        board.play_move(move, color)

        # current player is passing
        if move is None:
            break

    # get winner
    winner = GoBoardUtil.opponent(color)
    return winner
#################################################

class UCB:
    def __init__(self,sim_num,coefficient = 0.4):
        """
        NoGo player that selects moves according to
        flat Monte Carlo simulations with UCB.

        Parameters
        ----------
        name : str
            name of the player (used by the GTP interface).
        version : float
            version number (used by the GTP interface).
        """

        self.name = "UCB"
        self.version = 1.0
        self.sim = sim_num
        self.C = coefficient
        self.best_move = None
    
    
    ################ Getters & Setters #########################
    def set_sim_num(self, new_num):
        '''
        set new number of simulations
        '''
        self.sim = new_num
    
    def get_best_move(self):
        return self.best_move
    ############################################################
    
    ############### Core UCB Monte Carlo Logics ################
    def compute_ucb(self, num, val, N):
        '''
        calculate the upper confidence bound
        '''
        q = val/num
        return q + self.C*np.sqrt(np.log(N)/num)

    
    def select(self, stats, N):
        '''
        select the move to simulate based on the stats
        '''
        max_val = 0
        max_index = 0
        for index, (num, val) in enumerate(stats):
            # never selected so far
            if num == 0:
                return index
            # find the max ucb value and index
            ucb = self.compute_ucb(num, val, N)
            if ucb > max_val:
                max_val = ucb
                max_index = index
        
        return max_index
            
    def simulate(self, board:GoBoard, move, toplay):
        """
        Simulate a game for a given move.
        """
        cboard = board.copy()
        cboard.play_move(move, toplay)
        return play_game(cboard)
    
    def run_ucb(self, board:GoBoard, moves, color):
        '''
        Run the flat MC algorithm for N = #moves x #simulations times
        with UCB for move selection at each iteration.

        The move to act in the real game is the one with the max
        simulation count.
        '''
        total_sim = self.sim*len(moves)
        # first dimension: corresponding the moves
        # second dimension: [number of selection, total wins so far]
        stats = np.zeros((len(moves),2))

        for N in range(1, total_sim+1):
            # select move to simulate
            index = self.select(stats, N)
            move = moves[index]
            # simulate the game
            winner = self.simulate(board, move, color)
            if winner == color:
                # increment both countings
                stats[index] += 1
            else:
                # only increment number of selection
                stats[index][0] += 1
            
            # move index with maximum count
            max_index = np.argmax(stats,axis=0)[0]
            # update best move
            self.best_move = moves[max_index]

        return self.best_move
    
    ###############################################################

    def get_move(self, board:GoBoard, color:int):
        """
        Run one-ply MC simulations to get a move to play.
        """
        cboard = board.copy()
        cboard.current_player = color
        moves = GoBoardUtil.generate_legal_moves(cboard, color)

        # no legal moves left
        if not moves:
            return None
        # only one legal move to play, there is no other choice
        elif len(moves) == 1:
            return moves[0]
        # run ucb MC to determine the best move at present
        else:
            best = self.run_ucb(board, moves, color)
            return best
        
def run():
    """
    start the gtp connection and wait for commands.
    """
    board = GoBoard(7)
    con = GtpConnection(UCB(sim_num=100), board)
    con.start_connection()

if __name__ == "__main__":
    run()
