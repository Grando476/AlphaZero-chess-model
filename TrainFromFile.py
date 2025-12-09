import torch
import os
import random
from ChessGame import ChessNet
from train import load_self_play_data  

# Hyperparameters
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATA_DIR = "self_play_data_bestModel"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet(board_height=6, board_width=4).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
policy_loss_fn = torch.nn.CrossEntropyLoss()
value_loss_fn = torch.nn.MSELoss()

def train_on_file(file_path):
    print(f"\n Loading data from {file_path}...")
    states, policies, values = load_self_play_data(file_path)
    dataset_size = len(states)
    indices = list(range(dataset_size))

    model.train()
    for epoch in range(EPOCHS):
        random.shuffle(indices)
        total_policy_loss = 0
        total_value_loss = 0

        for start in range(0, dataset_size, BATCH_SIZE):
            batch_idx = indices[start:start+BATCH_SIZE]
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

        print(f"Epoch {epoch+1}/{EPOCHS} on {os.path.basename(file_path)} Policy Loss: {total_policy_loss:.4f}, Value Loss: {total_value_loss:.4f}")

# Main loop for all .pt files
for fname in sorted(os.listdir(DATA_DIR)):
    if fname.startswith("self_play_data_") and fname.endswith(".pt"):
        train_on_file(os.path.join(DATA_DIR, fname))

# Save final model
torch.save(model.state_dict(), "models/final_model_after_all_files3.pth")
print(" Model training complete and saved.")
