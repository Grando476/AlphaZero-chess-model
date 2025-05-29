# replay_buffer.py

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add_game(self, game_data):
        """
        game_data: list of (encoded_state, policy, z) tuples
        """
        self.buffer.append(game_data)

    def sample_batch(self, batch_size):
        """
        Sample full games, flatten into (state, policy, value) list
        """
        if batch_size > len(self.buffer):
            raise ValueError("Not enough games in buffer to sample")

        sampled_games = random.sample(self.buffer, batch_size)
        flat_batch = []
        for game in sampled_games:
            flat_batch.extend(game)
        return flat_batch

    def __len__(self):
        return len(self.buffer)
