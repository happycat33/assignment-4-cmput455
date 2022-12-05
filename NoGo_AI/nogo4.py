#!/usr/local/bin/python3
# /usr/bin/python3
# Set the path to your python3 above

#!/usr/bin/python3
# Set the path to your python3 above
from MCTS import MCTS
from gtp_connection import GtpConnection
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR, coord_to_point, opponent
from board import GoBoard
from engine import GoEngine

# Get the last move of the opponent
def find_opponent_move(last_board: GoBoard, board: GoBoard, opp_color: GO_COLOR):
    for row in range(1, board.size + 1):
        for col in range(1, board.size + 1):
            if last_board.get_color(last_board.pt(row, col)) \
                    != board.get_color(board.pt(row, col)) and \
                    board.get_color(board.pt(row, col)) == opp_color:
                return coord_to_point(row, col, board.size)
    assert False


class NoGo(GoEngine):
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
        self.mcts = MCTS()
        self.last_move = None
        self.last_board = None

    # Find one move using Monte Carlo Tree search
    def get_move(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        if self.last_board is not None:
            # find the move of the opponent
            opp_move = find_opponent_move(self.last_board, board, opponent(color))
        else:
            opp_move = None
        self.last_board = board.copy()

        # attempt to use previous subtree
        self.mcts.update_root(self.last_move, opp_move, color)

        # find the best move in the limited condition
        move = self.mcts.get_move(board, color)
        self.last_move = move
        return move

    # Get the best move so far
    def get_best_move(self):
        move = self.mcts.get_best_move()
        self.last_move = move
        return move


def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    con: GtpConnection = GtpConnection(NoGo(), board)
    con.start_connection()


if __name__ == "__main__":
    run()
