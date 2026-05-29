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
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):
        # x: (batch_size, a, b)，a=history_steps，b=pooled_seq_len
        x = x.unsqueeze(1).contiguous(memory_format=torch.channels_last) # Conv2d 用 NHWC，减少 layout 转换
        x = self.relu(self.conv1(x)) # (batch_size, 16, a, b)
        x = self.relu(self.conv2(x)) # (batch_size, 32, a, b)
        x = x.mean(dim=2, keepdim=True) # (batch_size, 32, 1, b)
        x = x.squeeze(2).contiguous() # Conv1d 输入保持 (batch_size, 32, b) 连续布局
        x = self.conv3(x)            # (batch_size, 1, b)
        x = x.squeeze(1)             # (batch_size, b)
        return x
