from typing import Dict, Tuple

import numpy as np

from board import GoBoard
from board_base import GO_COLOR, GO_POINT, NO_POINT, opponent, BLACK
from board_util import GoBoardUtil


# Calculate the uct value
def uct(child_wins: int, child_visits: int, parent_visits: int, exploration: float) -> float:
    return child_wins / child_visits + exploration * np.sqrt(np.log(parent_visits) / child_visits)


# It represents the node in the MCTS
class Node:
    """
    A node in the MCTS tree
    """

    def __init__(self, color: GO_COLOR) -> None:
        self.move: GO_POINT = NO_POINT
        self.color: GO_COLOR = color
        self.n_visits: int = 0
        self.n_opp_wins: int = 0
        self.parent: 'Node' = self
        self.children: Dict[Node] = {}
        self.expanded: bool = False

    # Set the parent of the node
    def set_parent(self, parent: 'Node') -> None:
        self.parent: 'Node' = parent

    def expand(self, board: GoBoard, color: GO_COLOR) -> None:
        """
        Expands tree by creating new children.
        """
        opp_color = opponent(board.current_player)
        moves = board.get_empty_points()
        for move in moves:
            if board.is_legal(move, color):
                # Create an new child
                node = Node(opp_color)
                node.move = move
                node.set_parent(self)
                self.children[move] = node

        self.expanded = True

    def select_in_tree(self, exploration: float) -> Tuple[GO_POINT, 'Node']:
        """
        Select move among children that gives maximizes UCT.
        If number of visits are zero for a node, value for that node is infinite, so definitely will get selected

        It uses: argmax(child_num_wins/child_num_vists + C * sqrt( ln(parent_num_vists) / child_num_visits )
        Returns:
        A tuple of (move, next_node)
        """
        chosen_child = None
        best_uct_val = -1
        for move, child in self.children.items():
            if child.n_visits == 0:
                # This child has not been visited, so it is chosen.
                return child.move, child

            # Attempt to find one child with the highest uct value
            uct_val = uct(child.n_opp_wins, child.n_visits, self.n_visits, exploration)
            if uct_val > best_uct_val:
                best_uct_val = uct_val
                chosen_child = child

        return chosen_child.move, chosen_child

    # Select the best one among the children
    def select_best_child(self) -> Tuple[GO_POINT, 'Node']:
        n_visits = -1
        best_child = None
        for move, child in self.children.items():
            if child.n_visits > n_visits:
                n_visits = child.n_visits
                best_child = child
        return best_child.move, best_child

    # Update the number of wins and visits
    def update(self, winner: GO_COLOR) -> None:
        self.n_opp_wins += self.color != winner
        self.n_visits += 1
        if not self.is_root():
            # Update the related values of the parent node
            self.parent.update(winner)

    # Check if the node is a leaf
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    # Check if the node is the root
    def is_root(self) -> bool:
        return self.parent == self

    # Calculate the number of all nodes
    def get_nodes_number(self):
        num = 1
        for child in self.children.values():
            num += child.get_nodes_number()

        return num

# It represents Monte Carlo Tree Search
class MCTS:
    # Initialize the Monte Carlo Tree Search
    def __init__(self):
        self.root: 'Node' = Node(BLACK)
        self.root.set_parent(self.root)
        self.exploration = 0.4

    # Run the MCTS to get one move for current player
    def get_move(self, board: GoBoard, color: int):
        """"
        Runs all playouts sequentially and returns the most visited move.
        """
        if not self.root.expanded:
            self.root.expand(board, color)

        for _ in range(100 * len(self.root.children)):
            # Perform the iteration of MCTS
            cboard = board.copy()
            self.single_playout(cboard, color)

        # choose a move that has the most visit
        best_move, _ = self.root.select_best_child()
        return best_move

    # Get the best move
    def get_best_move(self):
        best_move, _ = self.root.select_best_child()
        return best_move

    def single_playout(self, board: GoBoard, color: GO_COLOR) -> None:
        """
        Run a single playout from the root to the given depth, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        board -- a copy of the board.
        color -- color to play
        """
        # Choose one unexpanded node and then expand it
        node = self.root
        while not node.is_leaf():
            move, next_node = node.select_in_tree(self.exploration)
            assert board.play_move(move, color)
            color = opponent(color)
            node = next_node

        if not node.expanded:
            node.expand(board, color)

        # Perform the rollout
        assert board.current_player == color
        winner = self.rollout(board, color)

        # back propagation
        node.update(winner)

    def rollout(self, board: GoBoard, color: GO_COLOR) -> GO_COLOR:
        """
        Use the rollout policy to play until the end of the game
        """
        winner = play_game(board, color)
        return winner

    def update_root(self, last_move: GO_POINT, opp_move: GO_POINT, color: GO_COLOR) -> None:
        """
        Step forward in the tree, keeping everything we already know about the subtree, assuming
        that get_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            if opp_move in self.root.children:
                # Previous subtree can be used
                self.root = self.root.children[opp_move]
            else:
                self.root = Node(color)
        else:
            self.root = Node(color)

        self.root.set_parent(self.root)


def play_game(board: GoBoard, color: GO_COLOR):
    """
    Run a simulation game to the end from the current board
    """
    while True:
        # play a random move for the current player
        color = board.current_player
        move = GoBoardUtil.generate_random_move(board, color, False)

        # current player loses
        if move is None:
            break

        board.play_move(move, color)

    # get the winner
    winner = opponent(color)
    return winner
