# AlphaZero Chess Model

An implementation of the AlphaZero algorithm for chess, developed for educational and research purposes.

##  📖 Project Description

This repository hosts a Reinforcement Learning agent inspired by DeepMind's AlphaZero. Unlike traditional chess engines (like Stockfish) that rely on handcrafted evaluation functions, this model learns the game entirely from scratch through **self-play**. It combines **Monte Carlo Tree Search (MCTS)** with a deep residual neural network to estimate both move probabilities (policy) and position scores (value).

## ✨ Key Features

* **Self-Play Engine:** Robust pipeline for generating training data by pitting the current best model against itself.
* **Deep Neural Network:** A customized ResNet architecture (Policy & Value Network) trained on self-play data.
* **Iterative Improvement:** Automated evaluation capability to test new checkpoints against previous best versions.
* **User Interface:** Scripts to visualize the board and play interactively against the AI.
* **Configurable Hyperparameters:** Easy tuning of MCTS simulations, learning rate, and network depth via config files.

## 🛠️ Requirements

* **Python** 3.8+
* **PyTorch** (recommended) or TensorFlow
* **NumPy**
* **python-chess** (for move generation and validation)
* **Matplotlib** (for visualization, optional)

## 📂 File Structure

* `alpha_zero.py`: Core algorithm loop and orchestration.
* `mcts.py`: Efficient implementation of Monte Carlo Tree Search.
* `model.py`: Definition of the Residual Neural Network (ResNet).
* `train.py`: Training pipeline script.
* `eval.py`: Evaluation script for model comparison.
* `play.py`: Interface for Human vs. AI games.
* `config.py`: Configuration file for hyperparameters.
* `board_games`: **[University Project File]** -  File for the university assignment context.

* **`board games`** – This specific file is part of a university project and contains project-specific data/logic. Please keep this in mind when reviewing the code.

### How to play
Run play_vs_model.py to play versus specific model. To change the model you want to play against change the MODEL_PATH variable in play_vs_model.py file
