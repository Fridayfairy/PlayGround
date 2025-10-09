from re import M
import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        return x @ self.weight + self.bias

model = MyNet(3, 4)
print(model.parameters())
x = torch.randn(1, 3)
print(model(x))