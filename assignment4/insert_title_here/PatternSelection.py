# read in the weights from the file
import random

from board import GoBoard
from board_base import WHITE, BLACK, BORDER
from board_util import GoBoardUtil

weights = {}
with open('weights.txt', 'r') as fp:
    for line in fp.readlines():
        split = line.strip().split()
        index = int(split[0])
        weight = float(split[1])
        weights[index] = weight


def get_weight_of_move(board: GoBoard, move):
    # calculate the weight of the move based on pattern matching
    coord = point_to_coord(move, board.size)

    # left-top corner
    start_row, start_col = coord[0] - 1, coord[1] - 1

    # create the pattern
    pattern = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            row, col = start_row + i, start_col + j
            if 1 <= row <= board.size and 1 <= col <= board.size:
                color = board.get_color(board.pt(row, col))
            else:
                color = BORDER
            if row == coord[0] and col == coord[1]:
                pattern[i][j] = -1
            else:
                pattern[i][j] = color

    # get the base-4 address
    address = ''
    for i in range(3):
        for j in range(2, -1, -1):
            if pattern[i][j] == -1:
                continue

            address += str(pattern[i][j])

    # calculate the base-10 address
    final_address = 0
    for i in range(len(address)):
        idx = len(address) - 1 - i
        value = int(address[idx]) * 4 ** i
        final_address += value

    # get the weight
    weight = weights[final_address]

    return weight


def random_selection_with_probability(distribution):
    # select the move randomly according to the probabilities
    r = random.random()
    sum = 0.0
    for item in distribution:
        sum += item[1]
        if sum > r:
            return item[0]
    return distribution[-1][0]


def get_best_move_based_on_pattern(board: GoBoard, color: int):
    # get all legal moves
    moves = GoBoardUtil.generate_legal_moves(board, color)

    # calculate the weight for each legal move
    moves_list = []
    for move in moves:
        weight = get_weight_of_move(board, move)
        moves_list.append((move, weight))

    # calculate the probability of each move
    total_weight = sum([item[1] for item in moves_list])
    distribution = [(item[0], item[1] / total_weight) for item in moves_list]

    # select the move randomly according to the probabilities
    return random_selection_with_probability(distribution)


def point_to_coord(point, boardsize):
    """
    Transform point given as board array index
    to (row, col) coordinate representation.
    Special case: PASS is not transformed
    """
    NS = boardsize + 1
    return divmod(point, NS)


if __name__ == '__main__':
    # test
    board = GoBoard(3)
    board.board[board.pt(1, 1)] = BLACK
    board.board[board.pt(2, 1)] = WHITE
    board.board[board.pt(3, 1)] = WHITE
    board.board[board.pt(2, 3)] = BLACK
    board.board[board.pt(3, 3)] = BLACK
    print(str(GoBoardUtil.get_twoD_board(board)))
    print(point_to_coord(get_best_move_based_on_pattern(board, WHITE), board.size))
