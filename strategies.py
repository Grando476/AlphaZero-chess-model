from mcts import MCTS, MCTSNode
import torch
from utils import encode_state

class Strategy_MCTS:
    def __init__(self, game, model, simulations=200):
        self.game = game
        self.model = model
        self.simulations = simulations

    def choose_action(self, state, player):
        # Initialize root node
        root = MCTSNode(state, player)
        
        # Create MCTS tree with our model
        mcts = MCTS(self.game, self.model, simulations=self.simulations)
        mcts.run(root)

        # Choose action based on most visits
        visits = [(i, child.visits) for i, child in root.children.items()]
        if not visits:
            return None, 0  # No valid move
        best_action_index = max(visits, key=lambda x: x[1])[0]
        return best_action_index, root.children[best_action_index].value()

    def get_value(self, state, player):
        # Encode state and predict value directly from the model
        encoded = encode_state(state, player)
        input_tensor = torch.tensor(encoded, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(input_tensor)
        return value.item()
