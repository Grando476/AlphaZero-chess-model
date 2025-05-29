# collect_diverse_games.py
import os
import torch
from replay_buffer import ReplayBuffer
from self_play_top_board import self_play_game, game, model, strategy

# Settings
NUM_DIVERSE_GAMES = 20
DIVERSITY_SAVE_PATH = "diverse_games_buffer.pt"
REPLAY_BUFFER_SIZE = 100

# Create and fill replay buffer
buffer = ReplayBuffer(max_size=REPLAY_BUFFER_SIZE)

for game_id in range(NUM_DIVERSE_GAMES):
    print(f"Generating diversified game {game_id+1}/{NUM_DIVERSE_GAMES}")
    game_data = self_play_game(game, strategy, model, game_id)
    buffer.add_game(game_data)

# Save buffer
torch.save(buffer.buffer, DIVERSITY_SAVE_PATH)
print(f"Saved {len(buffer.buffer)} diversified games to {DIVERSITY_SAVE_PATH}")
