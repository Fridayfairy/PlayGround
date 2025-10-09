import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4))
        self.register_buffer("bias", torch.randn(4))
    
    def forward(self, x):
        return x @ self.weight + self.bias

model = MyNet()
print(model)
print("model.weight:", model.weight)
print("model.bias:", model.bias)
print("model.parameters():", list(model.parameters()))
print("model.buffers():", list(model.buffers()))

print("*"*88,"named_parameters")
for k, v in model.named_parameters():
    print(k, '\n', v, end="\n\n")
print("*"*88,"named_buffers")
for k, v in model.named_buffers():
    print(k, '\n', v, end="\n\n")