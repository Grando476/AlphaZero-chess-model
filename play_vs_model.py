import torch
from board_games_fun import Chess
from board_graphical_interface import Interface_Chess
from strategies import Strategy_MCTS
from ChessGame import ChessNet

# Paths
MODEL_PATH = "models/chessnet.pth"

# Setup
print("Loading model...")
model = ChessNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

game = Chess()
interface = Interface_Chess(game)
strategy = Strategy_MCTS(game, model, simulations=100)  # You can increase simulations for stronger AI

# Choose who starts: 1 - human starts (white), 2 - model starts (black)
HUMAN_PLAYER = 1

# Play game
interface.play_with_strategy(game, strategy, str_player=3-HUMAN_PLAYER)
