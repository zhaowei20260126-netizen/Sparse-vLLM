import torch
from torch import nn


class AttnPredictCNN(nn.Module):
    """Small CNN that predicts future attention distribution from historical attention patterns.

    Input: (batch_size, history_steps, pooled_seq_len) — already head-max-pooled
    Output: (batch_size, pooled_seq_len) — predicted importance score per block
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):
        # x: (batch_size, a, b) where a=history_steps, b=pooled_seq_len
        x = x.unsqueeze(1)           # (batch_size, 1, a, b)
        x = self.relu(self.conv1(x)) # (batch_size, 16, a, b)
        x = self.relu(self.conv2(x)) # (batch_size, 32, a, b)
        x = self.pool(x)             # (batch_size, 32, 1, b)
        x = x.squeeze(2)             # (batch_size, 32, b)
        x = self.conv3(x)            # (batch_size, 1, b)
        x = x.squeeze(1)             # (batch_size, b)
        return x