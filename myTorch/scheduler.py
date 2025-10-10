import torch 
import torch.optim as optim
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    )
optimizer = optim.Adam(model.parameters(),lr=0.01)
# StepLR, 没过10个epoch lr 乘以 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

inputs = torch.rand(10, 1)
targets = (inputs*2 + 1).detach()
criterion = nn.MSELoss()

for epoch in range(30):
    outputs = model(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Epoch: {epoch+1}, Loss:{loss.item()}, LR:{optimizer.param_groups[0]['lr']}")