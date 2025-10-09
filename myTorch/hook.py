import torch
import torch.nn as nn

features = {}

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, 1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, 1),
)

def save_feauture(name):
    def hook(module, input, output):
        features[name] = output
    return hook

hook_handle = model[0].register_forward_hook(save_feauture('ycxConv'))
hook_handle2 = model[2].register_forward_hook(save_feauture('ycxConv2'))

x = torch.randn(1, 3, 32, 32)
model(x)
print(features['ycxConv'].shape)
print(features['ycxConv2'].shape)
hook_handle.remove()
hook_handle2.remove()
