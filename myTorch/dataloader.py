import torch
from torch.utils.data import DataLoader, Dataset

class MyDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
data = torch.arange(1, 13).reshape(3, 4)
labels = data ** 2
datasets = MyDataSet(data, labels)
dataloader = DataLoader(datasets, batch_size=2, shuffle=True,num_workers=2)
for batch in dataloader:
    print(batch)