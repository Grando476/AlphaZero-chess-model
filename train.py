import torch
import torch.nn as nn
import torch.optim as optim
import os
from ChessGame import ChessNet

# Parameters
DATA_PATH = "self_play_data/self_play_data.pt"
SAVE_MODEL_PATH = "models/chessnet.pth"
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

def load_self_play_data(path):
    print(f"Loading self-play data from {path}")
    data = torch.load(path, weights_only=False)

    states = []
    policies = []
    values = []
    for state, policy, value in data:
        states.append(torch.tensor(state, dtype=torch.float32).permute(2,0,1))
        policies.append(torch.tensor(policy, dtype=torch.float32))
        values.append(torch.tensor([value], dtype=torch.float32))
    return states, policies, values

def train():
    # Load data
    states, policies, values = load_self_play_data(DATA_PATH)

    # Create model
    model = ChessNet()

    # Create optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Move everything to correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset_size = len(states)
    indices = list(range(dataset_size))

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        random.shuffle(indices)
        total_policy_loss = 0
        total_value_loss = 0

        for start_idx in range(0, dataset_size, BATCH_SIZE):
            batch_idx = indices[start_idx:start_idx+BATCH_SIZE]

            batch_states = torch.stack([states[i] for i in batch_idx]).to(device)
            batch_policies = torch.stack([policies[i] for i in batch_idx]).to(device)
            batch_values = torch.stack([values[i] for i in batch_idx]).to(device)

            optimizer.zero_grad()
            pred_policy_logits, pred_values = model(batch_states)

            policy_loss = policy_loss_fn(pred_policy_logits, batch_policies.argmax(dim=1))
            value_loss = value_loss_fn(pred_values.squeeze(), batch_values.squeeze())

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}: Policy Loss = {total_policy_loss:.4f}, Value Loss = {total_value_loss:.4f}")

    # Save trained model
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    train()
