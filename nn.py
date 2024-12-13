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

def load_ppg_raw_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(file_path)
    labels, data = [], []
    for i in df:
        labels.append(df[i][0]) # First row is the label
        data.append(df[i][1:]) # Second row onwards is the data
    return np.array(labels), np.array(data)


class PPGDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = np.empty(1)
        self.targets = np.empty(1)

    def __getitem__(self, index):
        # Fourni l'instance à un certain indice du jeu de données
        # Provide an instance of the dataset according to the index
        return self.data[index]

    def __len__(self):
        # Fournis la taille du jeu de données
        # Provide the lenght of the dataset
        return len(self.targets)

class PPGNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return False

if __name__ == "__main__":
    pass
