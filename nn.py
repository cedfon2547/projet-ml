import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from matplotlib import pyplot

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#########################################

class PPGDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.targets, self.data = [], []
        for _, d in pd.read_excel(path).items():
            self.targets.append(d[0]) # First row is the label
            self.data.append(d[1:])   # Second row onwards is the data
        self.targets, self.data = np.array(self.targets), np.array(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.targets)

class PPGNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return False

if __name__ == "__main__":
    ppg = PPGDataset("data/train8_reformat.xlsx")
    print(len(ppg))
