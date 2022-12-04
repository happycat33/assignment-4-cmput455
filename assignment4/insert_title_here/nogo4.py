#!/usr/local/bin/python3
# /usr/bin/python3
# Set the path to your python3 above

#!/usr/bin/python3
# Set the path to your python3 above



from gtp_connection import GtpConnection
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR, opponent
from PatternSelection import get_best_move_based_on_pattern
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine
from ucb import runUcb
from simulation_engine import GoSimulationEngine
from board_base import GO_COLOR, GO_POINT, BLACK, WHITE


class NoGo(GoSimulationEngine):
    def __init__(self, sim: int, move_select: str, sim_rule: str, 
                 check_selfatari: bool, limit: int = 100) -> None:
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
        GoSimulationEngine.__init__(self, "NoGo4", 1.0,
        sim, move_select, sim_rule, check_selfatari, limit)
        self.use_ucb = (move_select != "simple")
        self.use_pattern = (sim_rule == "rulebased")

    def simulate(self, board: GoBoard, move: GO_POINT,
                 toplay: GO_COLOR) -> GO_COLOR:
        """
        Run a simulated game for a given move.
        """
        cboard: GoBoard = board.copy()
        cboard.play_move(move, toplay)
        opp = opponent(toplay)
        return self.playGame(cboard, opp, komi=self.komi,
            limit=self.args.limit,
            random_simulation=self.args.sim_rule,
            use_pattern=self.use_pattern,
            check_selfatari=self.args.check_selfatari,)

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
def parse_args() -> Tuple[int, str, str, bool]:
    """
    Parse the arguments of the program.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--sim",
        type=int,
        default=10,
        help="number of simulations per move, so total playouts=sim*legal_moves",
    )
    parser.add_argument(
        "--moveselect",
        type=str,
        default="simple",
        help="type of move selection: simple or ucb",
    )
    parser.add_argument(
        "--simrule",
        type=str,
        default="prob",
        help="type of simulation policy: random or rulebased or prob",
    )
    parser.add_argument(
        "--movefilter",
        action="store_true",
        default=False,
        help="whether use move filter or not",
    )

    args = parser.parse_args()
    sim = args.sim
    move_select = args.moveselect
    sim_rule = args.simrule
    move_filter = args.movefilter

    if move_select != "simple" and move_select != "ucb":
        print("moveselect must be simple or ucb")
        sys.exit(0)
    if sim_rule != "random" and sim_rule != "rulebased" and sim_rule != "prob":
        print("simrule must be random or rulebased or prob")
        sys.exit(0)

    return sim, move_select, sim_rule, move_filter



def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    engine: NoGo = NoGo(sim, move_select, sim_rule, check_selfatari)
    con: GtpConnection = GtpConnection(engine, board)
    con.start_connection()


if __name__ == "__main__":
    sim, move_select, sim_rule, check_selfatari = parse_args()
    run(sim, move_select, sim_rule, check_selfatari)
