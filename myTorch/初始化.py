import torch
import torch.nn as nn
import torch.nn.init as init

class MyNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0.1)
        init.kaiming_uniform_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        

    def forward(self, x):
        return self.conv(x) @ self.weight + self.bias

