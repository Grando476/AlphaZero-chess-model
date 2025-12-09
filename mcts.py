import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from board_games_fun import Chess
import board_graphical_interface as bgra
import random
import matplotlib.pyplot as plt
from utils import encode_state


#def encode_state(state: Chess.ChessState, player: int) -> np.ndarray:
def encode_state(state: Chess.ChessState, player: int) -> np.ndarray:
    rows, cols = len(state.Board), len(state.Board[0])
    encoded = np.zeros((rows, cols, 12), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            piece = state.Board[row][col]
            if piece == 0:
                continue
            elif piece >= Chess.BlackShift:
                kind = piece - Chess.BlackShift
                channel = 6 + kind - 1
            else:
                kind = piece
                channel = kind - 1
            encoded[row, col, channel] = 1.0

    if player == 2:
        encoded = np.flip(encoded, axis=(0, 1)).copy()

    return encoded.copy()


# MCTS Node and Tree
class MCTSNode:
    def __init__(self, state, player, parent=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = 0.0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0


class MCTS:
    def __init__(self, game, model, simulations=50, c_puct=1.0):
        self.game = game
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, root):
        for _ in range(self.simulations):
            node = root
            search_path = [node]

            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            # Evaluation
            encoded = encode_state(node.state, node.player)
            input_tensor = torch.tensor(encoded.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = self.model(input_tensor)
            policy = torch.softmax(policy_logits, dim=1).numpy()[0]
            value = value.item()

            # Expansion
            legal_actions = self.game.actions(node.state, node.player)
            for i, action in enumerate(legal_actions):
                next_state, _ = self.game.next_state_and_reward(node.player, node.state, action)
                child_node = MCTSNode(next_state, 3 - node.player, parent=node)
                child_node.prior = policy[i] if i < len(policy) else 1.0 / len(legal_actions)
                node.children[i] = child_node

            # Backpropagation
            for node in reversed(search_path):
                node.visits += 1
                node.value_sum += value if node.player == root.player else -value

    def select_child(self, node):
        total_visits = sum(child.visits for child in node.children.values())
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            ucb = child.value() + self.c_puct * child.prior * (np.sqrt(total_visits) / (1 + child.visits))
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child


# Strategy Using MCTS and Neural Net
class Strategy_MCTS:
    def __init__(self, game, model, simulations=200):
        self.game = game
        self.model = model
        self.simulations = simulations

    def choose_action(self, state, player):
        root = MCTSNode(state, player)
        mcts = MCTS(self.game, self.model, simulations=self.simulations)
        mcts.run(root)

        # Pick the most visited child
        visits = [(i, child.visits) for i, child in root.children.items()]
        if not visits:
            return None, 0
        best_action_index = max(visits, key=lambda x: x[1])[0]
        return best_action_index, root.children[best_action_index].value()

    def get_value(self, state, player):
        encoded = encode_state(state, player)
        input_tensor = torch.tensor(encoded.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(input_tensor)
        return value.item()
