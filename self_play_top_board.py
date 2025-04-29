import torch
import numpy as np
import random
import os
import board_games_fun as bfun
from mcts import MCTSNode, MCTS
from ChessGame import ChessNet
from utils import encode_state

# Parameters
SIMULATIONS_PER_MOVE = 100
MAX_MOVES = 50  # Limit moves to avoid endless games

# Note: Action space still assumed to be 4672 for ChessNet compatibility

def self_play_game(game, strategy, model, game_id=0, max_moves=MAX_MOVES):
    """
    Play a single self-play game on custom board, return list of (state, pi, z)
    """
    data = []

    state = game.initial_state()
    player = 1
    step = 0
    game_over = False
    winner = None

    while not game_over:
        root = MCTSNode(state, player)
        mcts = MCTS(game, model, simulations=SIMULATIONS_PER_MOVE)
        mcts.run(root)

        legal_actions = game.actions(state, player)
        if not legal_actions:
            break

        visits = np.array([root.children.get(i, MCTSNode(None, None)).visits for i in range(len(legal_actions))])
        if visits.sum() == 0:
            action_probs = np.ones(len(visits)) / len(visits)
        else:
            action_probs = visits / visits.sum()

        # Save (encoded state, full 4672 policy vector, player)
        encoded = encode_state(state, player)
        full_policy = np.zeros(4672, dtype=np.float32)
        for idx in range(len(legal_actions)):
            full_policy[idx] = action_probs[idx]
        data.append((encoded, full_policy, player))

        # Choose move
        action_index = np.random.choice(len(action_probs), p=action_probs)
        action = legal_actions[action_index]

        next_state, reward = game.next_state_and_reward(player, state, action)

        # Step counter
        step += 1

        # End conditions
        if game.end_of_game(reward, step, state, action_index) or step >= max_moves:
            game_over = True
            if reward == 1:
                winner = 1
            elif reward == -1:
                winner = 2
            else:
                winner = 0

        player = 3 - player
        state = next_state

    # Assign final reward z to each move
    final_data = []
    for encoded, policy, move_player in data:
        if winner == 0:
            z = 0  # draw
        else:
            z = 1 if winner == move_player else -1
        final_data.append((encoded, policy, z))

    return final_data