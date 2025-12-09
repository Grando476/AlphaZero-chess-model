import torch
import os
import random
from ChessGame import ChessNet
from self_play_top_board import self_play_game  
from train import load_self_play_data
import board_games_fun as bfun
from strategies import Strategy_MCTS

# AlphaZero Loop Parameters
CYCLES = 20 # Number of self-play + training cycles
GAMES_PER_CYCLE = 100
SIMULATIONS_PER_MOVE = 60
SAVE_MODEL_PATH = "models/chessnet_top_board2.pth" #nie ruszać numeru 4 i 3!!!!!!
SAVE_DATA_DIR = "self_play_data_top"
BATCH_SIZE = 48
EPOCHS = 4
LEARNING_RATE = 0.003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure directories
os.makedirs("models", exist_ok=True)
os.makedirs(SAVE_DATA_DIR, exist_ok=True)

# Initialize model
model = ChessNet(board_height=6, board_width=4)


# Main loop
for cycle in range(CYCLES):
    print(f"=== Cycle {cycle+1}/{CYCLES} ===")

    # Self-play phase
    game = bfun.Chess("boards\szachy_plansza_top.txt")  
    strategy = Strategy_MCTS(game, model, simulations=SIMULATIONS_PER_MOVE)
    all_data = []

    for game_id in range(GAMES_PER_CYCLE):
        print(f"Playing self-play game {game_id+1}/{GAMES_PER_CYCLE}")
        data = self_play_game(game, strategy, model, game_id)
        all_data.extend(data)

    # Save collected data
    temp_data_path = os.path.join(SAVE_DATA_DIR, f"self_play_data_cycle{cycle+1}.pt")
    torch.save(all_data, temp_data_path)

    # Load data for training
    print("Loading self-play data for training...")
    states, policies, values = load_self_play_data(temp_data_path)
    dataset_size = len(states)
    indices = list(range(dataset_size))

    # Optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    # Training phase
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

    # Save model after training
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH} after cycle {cycle+1}")

print("AlphaZero Training Loop for custom board complete!")
