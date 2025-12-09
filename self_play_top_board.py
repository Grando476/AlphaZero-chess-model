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
SIMULATIONS_PER_MOVE = 80 # number of MCTS simulations per move
SAVE_DIR = "self_play_data"
MAX_MOVES = 40  # maximum number of moves per game
RANDOM_OPENING_MOVES = (0, 20)  # range of random opening moves
MATERIAL_REWARD_WEIGHT = 2  # multiplier for material advantage
CAPTURE_BIAS = 6  # capture action probability boost

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Initialize game and model
game = bfun.Chess("boards/szachy_plansza_top.txt")
model = ChessNet(board_height=6, board_width=4)

# Strategy
strategy = Strategy_MCTS(game, model, simulations=SIMULATIONS_PER_MOVE)

def apply_temperature(probs, temperature):
    if temperature <= 0:
        one_hot = np.zeros_like(probs)
        one_hot[np.argmax(probs)] = 1
        return one_hot
    adjusted = np.power(probs, 1.0 / temperature)
    adjusted /= np.sum(adjusted)
    return adjusted

def dynamic_temperature(step, max_steps):
    if step < max_steps * 0.3:
        return 3.5
    elif step < max_steps * 0.7:
        return 2
    else:
        return 1

def play_random_opening_moves(game, state, player, moves):
    for _ in range(moves):
        legal = game.actions(state, player)
        if not legal:
            break
        action = random.choice(legal)
        state, _ = game.next_state_and_reward(player, state, action)
        player = 3 - player
    return state, player

def material_value(piece):
    values = {'P': 1, 'S': 3, 'G': 3, 'W': 5, 'H': 9, 'p': 1, 's': 3, 'g': 3, 'w': 5, 'h': 9, }  
    return values.get(piece.upper(), 0)

def material_balance(state):
    white_total, black_total = 0, 0
    board_array = state.Board
    for row in board_array:
        for cell in row:
            if isinstance(cell, str):
                if cell.isupper():
                    white_total += material_value(cell)
                elif cell.islower():
                    black_total += material_value(cell)
    return white_total - black_total  # positive = white ahead, negative = black ahead

def is_capture_action(state, action):
    target_row, target_col = action[2], action[3]
    board = state.Board
    if 0 <= target_row < len(board) and 0 <= target_col < len(board[0]):
        target_cell = board[target_row][target_col]
        return isinstance(target_cell, str) and target_cell != '.'
    return False

def self_play_game(game, strategy, model, game_id=0, max_moves=MAX_MOVES):
    data = []
    state = game.initial_state()
    player = 1
    step = 0
    game_over = False
    winner = None
    ended_by_max_moves = False

    # Play some random opening moves to diversify starts
    opening_moves = random.randint(*RANDOM_OPENING_MOVES)
    state, player = play_random_opening_moves(game, state, player, opening_moves)

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

        # Apply capture bias before temperature
        for i, action in enumerate(legal_actions):
            if is_capture_action(state, action):
                action_probs[i] *= CAPTURE_BIAS
        action_probs /= np.sum(action_probs)

        # Apply dynamic temperature-based randomness
        temperature = dynamic_temperature(step, max_moves)
        action_probs = apply_temperature(action_probs, temperature)

        # Save full policy vector
        encoded = encode_state(state, player)
        full_policy = np.zeros(4672, dtype=np.float32)
        for idx in range(len(legal_actions)):
            full_policy[idx] = action_probs[idx]

        # Select action
        try:
            action_index = np.random.choice(len(action_probs), p=action_probs)
        except ValueError:
            break  # Skip game if no actions can be sampled safely

        action = legal_actions[action_index]

        before_material = material_balance(state)
        next_state, reward = game.next_state_and_reward(player, state, action)
        after_material = material_balance(next_state)

        # Reward based on material gain 
        material_diff = after_material - before_material if player == 1 else before_material - after_material
        material_reward = MATERIAL_REWARD_WEIGHT * material_diff

        data.append((encoded, full_policy, player, material_reward))

        step += 1
        if step == max_moves - 1:
            print(f"Step reached max_moves - 1: {step}")

        if game.end_of_game(reward, step, state, action_index) or step >= max_moves:
            game_over = True
            if reward == 1:
                winner = 1
            elif reward == -1:
                winner = 2
            else:
                winner = 0
            if step >= max_moves:
                ended_by_max_moves = True

        player = 3 - player
        state = next_state

    # Assign final reward z to each move
    final_data = []
    for encoded, policy, move_player, material_reward in data:
        if winner == 0:
            z = 0.5 if ended_by_max_moves else 0
        else:
            z = 15 if winner == move_player else -15
            if ended_by_max_moves:
                z *= 0.5
        z += material_reward  # combine with immediate material gain
        final_data.append((encoded, policy, z))

    return final_data

if __name__ == "__main__":
    # Self-play loop
    all_data = []
    max_move_ended = 0
    for game_id in range(NUM_GAMES):
        print(f"Playing game {game_id+1}/{NUM_GAMES}")
        game_data = self_play_game(game, strategy, model, game_id)
        if len(game_data) >= MAX_MOVES:
            max_move_ended += 1
        all_data.extend(game_data)

    # Save collected data
    save_path = os.path.join(SAVE_DIR, "self_play_data.pt")
    print(f"Saving self-play data to {save_path}")
    torch.save(all_data, save_path)

    print("Self-play completed!")
    print(f"Games ended due to reaching MAX_MOVES: {max_move_ended}/{NUM_GAMES}")
