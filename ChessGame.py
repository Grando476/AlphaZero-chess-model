import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        # We assume input shape: (12 channels, 8 rows, 8 cols)

        # Common convolutional body
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)  # 4 channels
        self.policy_fc = nn.Linear(4 * 8 * 8, 4672)  # 4672 = 8x8x73 possible moves (simplified output space)

        # Value head
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)  # 2 channels
        self.value_fc1 = nn.Linear(2 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, 12, 8, 8)

        # Common body
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)  # flatten
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # value between -1 and 1

        return p, v

# Example usage
if __name__ == "__main__":
    net = ChessNet()
    dummy_input = torch.randn(1, 12, 8, 8)  # batch of 1 board
    policy_logits, value = net(dummy_input)
    print("Policy logits shape:", policy_logits.shape)
    print("Value shape:", value.shape)