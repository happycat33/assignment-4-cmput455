#!/usr/local/bin/python3
# /usr/bin/python3
# Set the path to your python3 above

#!/usr/bin/python3
# Set the path to your python3 above



from gtp_connection import GtpConnection
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR
from PatternSelection import get_best_move_based_on_pattern
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine
from ucb import runUcb
from board_base import GO_COLOR, GO_POINT, BLACK, WHITE


class NoGo:
    def __init__(self):
        """
        Go player that selects moves randomly from the set of legal moves.
        Does not use the fill-eye filter.
        Passes only if there is no other legal move.

        Parameters
        ----------
        name : str
            name of the player (used by the GTP interface).
        version : float
            version number (used by the GTP interface).
        """
        GoEngine.__init__(self, "NoGo4", 1.0)

    def simulate(self, board: GoBoard, move: GO_POINT,
                 toplay: GO_COLOR) -> GO_COLOR:
        """
        Run a simulated game for a given move.
        """
        cboard: GoBoard = board.copy()
        cboard.play_move(move, toplay)
        return self.playGame(cboard)

    def playGame(self, board: GoBoard) -> GO_COLOR:
        """
        Run a simulation game.
        """
        winner = self.get_winner(board)
        while winner is None:
            color = board.current_player
            if self.args.random_simulation:
                # random policy
                move = GoBoardUtil.generate_random_move(board, color, False)
            else:
                # Pattern-based probabilistic
                move = get_best_move_based_on_pattern(board, color)

            board.play_move(move, color)

            # check if the game is over
            winner = self.get_winner(board)

        return winner

    def get_moves(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """
        Run one-ply MC simulations to get a move to play.
        """
        cboard = board.copy()
        legal_moves = GoBoardUtil.generate_legal_moves(cboard, color)

        if self.args.use_ucb:
            # UCB
            C = 0.4  # sqrt(2) is safe, this is more aggressive
            moves = runUcb(self, cboard, C, legal_moves, color)
            return moves
        else:
            # Round Robin
            moveWins = []
            for move in legal_moves:
                wins = self.simulateMove(cboard, move, color)
                moveWins.append(wins)
            total = sum(moveWins)
            return [(legal_moves[i], 0 if total == 0 else moveWins[i] / total)
                    for i in
                    range(len(legal_moves))]

    def genmove(self, state):
        assert not state.endOfGame()
        moves = state.legalMoves()
        numMoves = len(moves)
        score = [0] * numMoves
        for i in range(numMoves):
            move = moves[i]
            score[i] = self.simulate(state, move)
        #print(score)
        bestIndex = score.index(max(score))
        best = moves[bestIndex]
        #print("Best move:", best, "score", score[best])
        assert best in state.legalMoves()
        return best

    def get_winner(self, board: GoBoard):
        # get current winner
        legal_moves = GoBoardUtil.generate_legal_moves(board,
                                                       board.current_player)
        if len(legal_moves) > 0:
            return None
        else:
            if board.current_player == BLACK:
                return WHITE
            else:
                return BLACK



def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    con: GtpConnection = GtpConnection(NoGo(), board)
    con.start_connection()


if __name__ == "__main__":
    run()
