class Strategy_MCTS:
    def __init__(self, game, model, simulations=50):
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
        input_tensor = torch.tensor(encoded).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(input_tensor)
        return value.item()