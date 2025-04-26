import torch
import numpy as np
import random
import os
from board_games_fun import Chess
from mcts import MCTSNode, MCTS
from ChessGame import ChessNet
from utils import encode_state
from strategies import Strategy_MCTS
import board_games_fun as bfun

# Parameters
NUM_GAMES = 2  # number of self-play games
SIMULATIONS_PER_MOVE = 100  # number of MCTS simulations per move
SAVE_DIR = "self_play_data"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Initialize game and model
game = bfun.Chess("boards\szachy_plansza_top.txt")

model = ChessNet()

# Strategy
strategy = Strategy_MCTS(game, model, simulations=SIMULATIONS_PER_MOVE)

def self_play_game(game, strategy, model, game_id=0, max_moves=10):

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

        visits = np.array([root.children.get(i, MCTSNode(None, None)).visits for i in range(len(game.actions(state, player)))])
        if visits.sum() == 0:
            action_probs = np.ones(len(visits)) / len(visits)
        else:
            action_probs = visits / visits.sum()

        # Save (state, policy)
                # Save (state, full policy vector)
        encoded = encode_state(state, player)

        # Create a 4672-dimensional policy vector
        full_policy = np.zeros(4672, dtype=np.float32)
        legal_actions = game.actions(state, player)
        
        for idx, action in enumerate(legal_actions):
            full_policy[idx] = action_probs[idx]  # assuming first N outputs correspond to legal actions

        data.append((encoded, full_policy, player))


        # Select action
        action_index = np.random.choice(len(action_probs), p=action_probs)
        actions = game.actions(state, player)
        action = actions[action_index]

        next_state, reward = game.next_state_and_reward(player, state, action)

        # Update step counter
        step += 1
        print(step)
        # Check end of game
        if game.end_of_game(reward, step, state, action_index) or step >= max_moves:
            game_over = True
            if reward == 1:
                winner = 1
            elif reward == -1:
                winner = 2
            else:
                winner = 0  # Draw if max moves reached

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


# Self-play loop
all_data = []
for game_id in range(NUM_GAMES):
    print(f"Playing game {game_id+1}/{NUM_GAMES}")
    game_data = self_play_game(game, strategy, model, game_id)
    all_data.extend(game_data)

# Save collected data
save_path = os.path.join(SAVE_DIR, "self_play_data.pt")
print(f"Saving self-play data to {save_path}")
torch.save(all_data, save_path)

print("Self-play completed!")
