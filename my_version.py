import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from board_games_fun import Chess
import board_graphical_interface as bgra
import random
import matplotlib.pyplot as plt
from mcts import MCTS, MCTSNode, Strategy_MCTS 
from utils import encode_state

# === State Encoding Function ===



# === Neural Network Placeholder ===
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessPolicyValueNet(nn.Module):
    def __init__(self, rows=8, cols=8):
        super().__init__()
        self.rows = rows
        self.cols = cols
        flat_size = rows * cols

        self.input_block = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(2 * flat_size, flat_size * flat_size)

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc1 = nn.Linear(flat_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.input_block(x)
        x = self.res_blocks(x)

        # Policy
        p = self.policy_head(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = self.value_head(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


# === Strategy Wrapper Using Neural Net ===
class Strategy_Neural:
    def __init__(self, game_obj, model: ChessPolicyValueNet):
        self.game_obj = game_obj
        self.model = model
        self.model.eval()

    def choose_action(self, state, player):
        encoded = encode_state(state, player)
        input_tensor = torch.tensor(encoded.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.model(input_tensor)
        policy = torch.softmax(policy_logits, dim=1).numpy()[0]
        legal_actions = self.game_obj.actions(state, player)
        if not legal_actions:
            return None, value.item()

        policy_slice = policy[:len(legal_actions)]
        if np.any(np.isnan(policy_slice)) or np.sum(policy_slice) == 0:
            action_idx = np.random.randint(len(legal_actions))
        else:
            policy_slice /= np.sum(policy_slice)
            action_idx = np.random.choice(len(legal_actions), p=policy_slice)

        return action_idx, value.item()

    def get_value(self, state, player):
        encoded = encode_state(state, player)
        input_tensor = torch.tensor(encoded).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(input_tensor)
        return value.item()

def play_game_with_model(game_obj, model, str_player=1):
    interface = bgra.Interface_Chess(game_obj)
    strategy = Strategy_MCTS(game_obj, model)
    interface.play_with_strategy(game_obj, strategy, str_player)

# === Self-Play Loop ===
def self_play(game, model, num_games=10, simulations=50):
    data = []

    for _ in range(num_games):
        state = game.initial_state()
        player = 1
        trajectory = []
        done = False

        while not done:
            strategy = Strategy_MCTS(game, model, simulations)
            encoded = encode_state(state, player)

            legal_actions = game.actions(state, player)
            if not legal_actions:
                break

            action_idx, _ = strategy.choose_action(state, player)
            if action_idx is None:
                break

            rows, cols = len(state.Board), len(state.Board[0])  # Moved here before action
            flat_size = rows * cols

            action = legal_actions[action_idx]
            next_state, reward = game.next_state_and_reward(player, state, action)

            policy_vector = np.zeros(flat_size * flat_size, dtype=np.float32)

            try:
                fr, fc, tr, tc, *_ = action

                if not (0 <= fr < rows and 0 <= fc < cols and 0 <= tr < rows and 0 <= tc < cols):
                    print("[WARNING] Skipping invalid board coordinates:", action)
                    continue

                from_index = fr * cols + fc
                to_index = tr * cols + tc
                action_index = from_index * flat_size + to_index

                if action_index < policy_vector.size:
                    policy_vector[action_index] = 1.0
                else:
                    print("[WARNING] Skipping out-of-bounds action index:", action_index)
                    continue

            except Exception as e:
                print("ACTION ERROR:", action)
                raise e

            trajectory.append((encoded, policy_vector, player))

            done = game.end_of_game(reward, 0, next_state, action_idx)
            state = next_state
            player = 3 - player

        final_value = 1 if reward == 1 else -1 if reward == -1 else 0
        for encoded, policy, who in trajectory:
            value = final_value if who == 1 else -final_value
            data.append((encoded, policy, value))

    return data



def train_model(model, data, epochs=25, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        for state, policy_target, value_target in data:
            x = torch.tensor(state.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            policy_target = torch.tensor(policy_target, dtype=torch.float32).unsqueeze(0)  # shape: [1, N]
            value_target = torch.tensor([[value_target]], dtype=torch.float32)  # shape: [1, 1]

            pred_policy, pred_value = model(x)

            # Apply log_softmax to predicted policy for KLDivLoss
            pred_policy_log = F.log_softmax(pred_policy, dim=1)
            loss_policy = F.kl_div(pred_policy_log, policy_target, reduction="batchmean")

            loss_value = F.mse_loss(pred_value, value_target)
            loss = loss_policy + loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()

        print(f"Epoch {epoch+1}: Total Loss = {total_loss:.4f}, Policy Loss = {total_policy_loss:.4f}, Value Loss = {total_value_loss:.4f}")



# === Evaluation Function ===
def evaluate_models(game, model_new, model_old, games=8):
    wins_new = 0
    wins_old = 0
    draws = 0
    
    for i in range(games):
        move_count = 0
        max_moves = 100
        print(i)
        print(" games played")
        state = game.initial_state()
        player = 1
        done = False

        while not done and move_count < max_moves:
            strategy = Strategy_MCTS(game, model_new if player == 1 else model_old)
            action_idx, _ = strategy.choose_action(state, player)
            legal_actions = game.actions(state, player)
            if not legal_actions or action_idx is None:
                break
            action = legal_actions[action_idx]
            state, reward = game.next_state_and_reward(player, state, action)
            done = game.end_of_game(reward, 0, state, action_idx)
            player = 3 - player
            move_count += 1

        if reward == 1:
            wins_new += 1
        elif reward == -1:
            wins_old += 1
        else:
            draws += 1

    print(f"Evaluation: new={wins_new}, old={wins_old}, draws={draws}")
    return wins_new > wins_old


# === Training Loop with Comparison ===
def train_loop(game, model_path="best_model.pth", iterations=1, games_per_iteration=10):
    rows, cols = len(game.initial_state().Board), len(game.initial_state().Board[0])
    try:
        from my_version import ChessPolicyValueNet
        model_best = ChessPolicyValueNet(rows, cols)
        model_best.load_state_dict(torch.load(model_path))

        print(" Loaded existing model")
    except:
        from my_version import ChessPolicyValueNet
        model_best = ChessPolicyValueNet(rows, cols)
        print(" Created new model")

    for i in range(iterations):
        print(f"Iteration {i+1}: Generating self-play games...")
        data = self_play(game, model_best, num_games=games_per_iteration)

        from my_version import ChessPolicyValueNet
        model_new = ChessPolicyValueNet(rows, cols)
        model_new.load_state_dict(model_best.state_dict())
        train_model(model_new, data)

        print(" Evaluating new model...")
        if evaluate_models(game, model_new, model_best):
            torch.save(model_new.state_dict(), model_path)
            model_best = model_new
            print(" New model accepted and saved")
        else:
            print(" New model rejected")


# === Main Run Block ===
if __name__ == "__main__":
    game = Chess("boards\szachy_plansza_top.txt")
    train_loop(game, iterations=1, games_per_iteration=5)
    print(" Training complete. You can now play against the model using the GUI.")

    from my_version import ChessPolicyValueNet
    rows, cols = len(game.initial_state().Board), len(game.initial_state().Board[0])
    model = ChessPolicyValueNet(rows, cols)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    print("Entering interactive play loop vs AI...")
    while True:
        play_game_with_model(game, model, str_player=1)
        again = input("Play again? (y/n): ").strip().lower()
        if again != 'y':
            print("Exiting game loop. Goodbye!")
            break