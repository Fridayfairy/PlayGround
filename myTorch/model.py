import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

model = MyNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
inputs = torch.rand(10, 3)
targets = (inputs*2 + 1).detach()

for epoch in range(1000):
    outputs = model(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss:{loss.item()}")

with torch.no_grad():
    print("Target:", targets)
    print("Output:", model(inputs))