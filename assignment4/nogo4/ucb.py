# Cmput 455 sample code
# UCB algorithm
# Written by Martin Mueller

from numpy import signedinteger, intc
from numpy.typing import _32Bit

from board_base import GO_COLOR, GO_POINT, NO_POINT, PASS, MAXSIZE
from board import GoBoard


from board_base import GO_COLOR, GO_POINT, NO_POINT, PASS
from board import GoBoard
from simulation_engine import GoSimulationEngine

import sys
from math import log, sqrt

from typing import List, Tuple, Union, Any

INFINITY = float("inf")

STATS = List[List[int]]
# tuple = (move, percentage, wins, pulls)
TUPLE = Tuple[str, float, int, int]

def point_to_coord(point, boardsize):
    """
    Transform point given as board array index
    to (row, col) coordinate representation.
    Special case: PASS is not transformed
    """
    if point == PASS:
        return PASS
    else:
        NS = boardsize + 1
        return divmod(point, NS)

def format_point(move):
    """
    Return move coordinates as a string such as 'A1', or 'PASS'.
    """
    assert MAXSIZE <= 25
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if move == PASS:
        return "PASS"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1] + str(row)

def mean(stats: STATS, i: int) -> float:
    return stats[i][0] / stats[i][1]


def ucb(stats: STATS, C: float, i: int, n: int) -> float:
    if stats[i][1] == 0:
        return INFINITY
    return mean(stats, i) + C * sqrt(log(n) / stats[i][1])


def findBest(stats: STATS, C: float, n: int) -> int:
    best = -1
    bestScore = -INFINITY
    for i in range(len(stats)):
        score = ucb(stats, C, i, n)
        if score > bestScore:
            bestScore = score
            best = i
    assert best != -1
    return best


def bestArm(stats: STATS) -> int:  # Most-pulled arm
    best = -1
    bestScore = -INFINITY
    for i in range(len(stats)):
        if stats[i][1] > bestScore:
            bestScore = stats[i][1]
            best = i
    assert best != -1
    return best


def byPercentage(tuple: TUPLE) -> float:
    return tuple[1]

def byPulls(tuple: TUPLE) -> float:
    return tuple[3]

def writeMoves(board: GoBoard, moves: List[GO_POINT], stats: STATS) -> None:
    gtp_moves = []
    for i in range(len(moves)):
        if moves[i] != PASS:
            x, y = point_to_coord(moves[i], board.size)
            pointString = format_point((x, y))
        else:
            pointString = "Pass"
        if stats[i][1] != 0:
            gtp_moves.append(
                (pointString, stats[i][0] / stats[i][1], stats[i][0], stats[i][1])
            )
        else:
            gtp_moves.append((pointString, 0.0, stats[i][0], stats[i][1]))
    sys.stderr.write(
        "Statistics: {}\n".format(sorted(gtp_moves, key=byPulls, reverse=True))
    )
    sys.stderr.flush()


def runUcb(player: GoSimulationEngine, board: GoBoard, C: float, 
    moves: List[GO_POINT], toplay: GO_COLOR) -> GO_POINT:
    stats = [[0, 0] for _ in moves]
    num_simulation = len(moves) * player.args.sim
    for n in range(num_simulation):
        moveIndex = findBest(stats, C, n)
        result = player.simulate(board, moves[moveIndex], toplay)
        if result == toplay:
            stats[moveIndex][0] += 1  # win
        stats[moveIndex][1] += 1

    total = sum([stats[i][0] for i in range(len(moves))])
    return [(moves[i], 0 if total == 0 else stats[i][0] / total) for i in range(len(moves))]
