import time
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#########################################

class PPGDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data, self.targets = [], []
        for _, d in pd.read_excel(path).items():
            self.data.append([d[1:]])
            self.targets.append(d[0])
        self.data = np.array(self.data).astype(np.float32)
        self.targets = np.array(self.targets) - 1 # Labels start 1, which is class 0

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

class PPGNetv1(nn.Module):
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

        self.S6 = nn.AvgPool1d(4) # input: [N, 32, 8]
        self.L7 = nn.Linear(32*2, 35)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        y = F.relu(self.C1_norm(self.C1(x)))
        y = F.relu(self.C2_norm(self.C2(y)))
        y = F.relu(self.C3_norm(self.C3(y)))
        y = F.relu(self.C4_norm(self.C4(y)))
        y = self.S6(self.C5_norm(self.C5(y))) # output: [N, 32, 2]
        y = y.view(y.size(0), -1)             # output: [N, 64]
        return self.output(self.L7(y))        # output: [N, 35]

def train_eval(model, train_set, test_set, epochs=10, lr=0.01, mom=0.9, bs=32, show=True):
    # preparations
    model.to(DEVICE)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=mom)

    # training
    print(f"=== Training {model.__class__.__name__} ===")
    model.train()
    for i_epoch in range(epochs):
        start_time, train_losses = time.time(), []
        for signals, targets in train_loader:
            signals = signals.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()

            predictions = model(signals)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

        print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
            i_epoch+1, epochs, np.mean(train_losses), time.time()-start_time))

    # evaluation
    print(f"\n=== Evaluating {model.__class__.__name__} ===")
    model.eval()
    all_predictions, all_targets = [], []

    for signals, targets in test_loader:
        signals = signals.to(DEVICE)
        targets = targets.to(DEVICE)
        with torch.no_grad():
            predictions = torch.argmax(model(signals), dim=1)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    cmatrix = confusion_matrix(np.concatenate(all_predictions), np.concatenate(all_targets))
    accuracy = np.trace(cmatrix)/np.sum(cmatrix)

    # results
    if show:
        print(f"Accuracy: {accuracy:.4f}\n")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cmatrix, annot=True, fmt='d')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix for PPGNetv1')
        plt.show()

    return cmatrix, accuracy

if __name__ == "__main__":
    train_set = PPGDataset("data/train8_reformat.xlsx")
    test_set = PPGDataset("data/test8_reformat.xlsx")

    train_eval(PPGNetv1(), train_set, test_set, epochs=100, bs=300)
