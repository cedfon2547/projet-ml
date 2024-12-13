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
        self.C1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, bias=False)
        self.C1_norm = nn.BatchNorm1d(16)

        self.C2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, bias=False)
        self.C2_norm = nn.BatchNorm1d(32)

        self.C3 = nn.Conv1d(32, 32, kernel_size=3, stride=2, bias=False)
        self.C3_norm = nn.BatchNorm1d(32)

        self.C4 = nn.Conv1d(32, 32, kernel_size=3, stride=2, bias=False)
        self.C4_norm = nn.BatchNorm1d(32)

        self.C5 = nn.Conv1d(32, 32, kernel_size=3, stride=2, bias=False)
        self.C5_norm = nn.BatchNorm1d(32)
        #should be size 8
        print(self.C5_norm.shape)
        self.S6 = nn.AvgPool1d(2)
        print(self.S6.shape)
        # out is 7

        self.L7 = nn.Linear(32*7, 35)

        self.output = nn.Sigmoid()

    def forward(self, x):
        y = F.relu(self.C1_norm(self.C1(x)))
        y = F.relu(self.C2_norm(self.C2(y)))
        y = F.relu(self.C3_norm(self.C3(y)))
        y = F.relu(self.C4_norm(self.C4(y)))
        y = self.C5_norm(self.C5(y))
        y = self.S6(y)
        y = y.view(y.size(0), -1)
        y = self.L7(y)
        y = self.L8(y)
        return self.output(y)

if __name__ == "__main__":
    ppg = PPGDataset("data/train8_reformat.xlsx")
    print(len(ppg))
