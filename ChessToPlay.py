from board_games_fun import Chess
from board_graphical_interface import Interface_Chess
from strategies import Strategy_MCTS
from ChessGame import ChessNet  # nasz plik z siecią!

# Setup
game = Chess()
model = ChessNet()
interface = Interface_Chess(game)
strategy = Strategy_MCTS(game, model, simulations=100)

# Play
interface.play_with_strategy(game, strategy, str_player=2)  # graj jako białe przeciwko AI
