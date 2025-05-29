import torch
import random
import os
from board_games_fun import Chess
from mcts import MCTSNode, MCTS
from ChessGame import ChessNet
from utils import encode_state
from strategies import Strategy_MCTS
from replay_buffer import ReplayBuffer

# Parameters
SIMULATIONS_PER_MOVE = 130
MAX_MOVES = 40
REPLAY_BUFFER_PATH = "self_play_data/replay_buffer.pt"
GAMES_PER_LOOP = 5

# Load model
game = Chess("boards/szachy_plansza_top.txt")
model = ChessNet(board_height=6, board_width=4)
model.load_state_dict(torch.load("models/chessnet_top_board.pth", map_location=torch.device('cpu')))
model.eval()

strategy = Strategy_MCTS(game, model, simulations=SIMULATIONS_PER_MOVE)

# Load replay buffer
if os.path.exists(REPLAY_BUFFER_PATH):
    print(f"Loading replay buffer from {REPLAY_BUFFER_PATH}")
    replay_buffer = torch.load(REPLAY_BUFFER_PATH)
else:
    raise FileNotFoundError("Replay buffer file not found. Run 'collect_diverse_games.py' first.")

# Function to simulate self-play from selected games
def replay_and_train(game, model, strategy, replay_buffer, num_games):
    selected_games = replay_buffer.sample_games(num_games)
    new_training_data = []

    for i, game_data in enumerate(selected_games):
        print(f"Replaying game {i+1}/{num_games}")
        for encoded, policy, z in game_data:
            new_training_data.append((encoded, policy, z))

    # Save the new data from replay
    save_path = os.path.join("self_play_data", "self_play_replayed.pt")
    print(f"Saving replayed self-play data to {save_path}")
    torch.save(new_training_data, save_path)
    return new_training_data

if __name__ == "__main__":
    replay_and_train(game, model, strategy, replay_buffer, GAMES_PER_LOOP)
    print("Replay self-play completed.")
