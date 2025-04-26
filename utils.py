import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from board_games_fun import Chess
import board_graphical_interface as bgra
import random
import matplotlib.pyplot as plt

def encode_state(state: Chess.ChessState, player: int) -> np.ndarray:
    rows, cols = len(state.Board), len(state.Board[0])
    encoded = np.zeros((rows, cols, 12), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            piece = state.Board[row][col]
            if piece == 0:
                continue
            elif piece >= Chess.BlackShift:
                kind = piece - Chess.BlackShift
                channel = 6 + kind - 1
            else:
                kind = piece
                channel = kind - 1
            encoded[row, col, channel] = 1.0

    if player == 2:
        encoded = np.flip(encoded, axis=(0, 1))

    return encoded