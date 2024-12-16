import time
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from scipy import signal
# from sklearn.preprocessing import MinMaxScaler

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#########################################
def load_ppg_raw_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(file_path)
    labels, data = [], []
    for i in df:
        labels.append(df[i][0]) # First row is the label
        data.append(df[i][1:]) # Second row onwards is the data
    return np.array(labels), np.array(data)

def get_peaks(ppg, fs):
    return signal.find_peaks(ppg, distance=fs / 2)[0]

def segment_ppg(data, peaks):
    segments = []
    for i in range(len(peaks) - 1):
        segments.append(data[peaks[i]:peaks[i + 1]])
    return segments

class PPGDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        # scaler = MinMaxScaler()
        # sos = signal.butter(3, [0.004, 8], fs=50, btype='band', output='sos')
        self.data, self.targets = [], []
        for _, d in pd.read_excel(path).items():
            t = np.array(d[1:])
            # t = scaler.fit_transform(t.reshape(-1,1)).flatten()
            # t = signal.sosfilt(sos, t)
            # plt.figure()
            # plt.plot(np.arange(300), t)
            # plt.show()
            self.data.append([t])
            self.targets.append(d[0])
        self.data = np.array(self.data).astype(np.float32)
        self.targets = np.array(self.targets) - 1 # Labels start 1, which is class 0

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

class PPGDatasetSegments(Dataset):
    def __init__(self, path):
        super().__init__()
        labels, data = load_ppg_raw_data(path)

        fs = 50  # Sampling frequency
        segments_list = []
        segment_labels = []

        for i in range(len(data)):
            peaks = get_peaks(data[i], fs=fs)
            segments = segment_ppg(data[i], peaks)
            segments_list.extend(segments)
            segment_labels.extend([labels[i]] * len(segments))

        self.data = segments_list
        self.targets = segment_labels
        print(len(self.data))
        print(len(self.targets))

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

class PPGNetv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, bias=False)
        self.N1 = nn.BatchNorm1d(64)

        self.C2 = nn.Conv1d(64, 64, kernel_size=3, stride=2, bias=False)
        self.N2 = nn.BatchNorm1d(64)

        self.C3 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N3 = nn.BatchNorm1d(64)

        self.C4 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N4 = nn.BatchNorm1d(64)

        self.C5 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N5 = nn.BatchNorm1d(64)

        self.C6 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N6 = nn.BatchNorm1d(64)

        self.C7 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N7 = nn.BatchNorm1d(64)

        self.L8 = nn.Linear(256, 128)
        self.N8 = nn.BatchNorm1d(128)

        self.L9 = nn.Linear(128, 64)
        self.N9 = nn.BatchNorm1d(64)

        self.D10 = nn.Dropout1d(0.1)
        self.L10 = nn.Linear(64, 35)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        y = F.relu(self.N1(self.C1(x)))
        y = F.relu(self.N2(self.C2(y)))
        y = F.relu(self.N3(self.C3(y)))
        y = F.relu(self.N4(self.C4(y)))
        y = F.relu(self.N5(self.C5(y)))
        y = F.relu(self.N6(self.C6(y)))
        y = F.relu(self.N7(self.C7(y)))

        y = y.view(y.size(0), -1)
        y = F.relu(self.N8(self.L8(y)))
        y = F.relu(self.N9(self.L9(y)))

        return self.output(self.L10(self.D10(y)))

class PPGNetv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv1d(1, 64, kernel_size=5, stride=3, bias=False)
        self.N1 = nn.BatchNorm1d(64)

        self.C2 = nn.Conv1d(64, 64, kernel_size=3, stride=2, bias=False)
        self.N2 = nn.BatchNorm1d(64)

        self.C3 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N3 = nn.BatchNorm1d(64)

        self.C4 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N4 = nn.BatchNorm1d(64)

        self.C5 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N5 = nn.BatchNorm1d(64)

        self.C6 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N6 = nn.BatchNorm1d(64)

        self.C7 = nn.Conv1d(64, 64, kernel_size=2, stride=2, bias=False)
        self.N7 = nn.BatchNorm1d(64)

        self.L8 = nn.Linear(64, 16)
        self.N8 = nn.BatchNorm1d(16)

        self.L9 = nn.Linear(16, 64)
        self.N9 = nn.BatchNorm1d(64)

        self.L10 = nn.Linear(64, 64)
        self.N10 = nn.BatchNorm1d(64)

        self.L11 = nn.Linear(64, 64)
        self.N11 = nn.BatchNorm1d(64)

        self.D12 = nn.Dropout1d(0.1)
        self.L12 = nn.Linear(64, 35)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        y = F.relu(self.N1(self.C1(x)))
        y = F.relu(self.N2(self.C2(y)))
        y = F.relu(self.N3(self.C3(y)))
        y = F.relu(self.N4(self.C4(y)))
        y = F.relu(self.N5(self.C5(y)))
        y = F.relu(self.N6(self.C6(y)))
        y = F.relu(self.N7(self.C7(y)))
        y = y.view(y.size(0), -1)
        y = F.relu(self.N8(self.L8(y)))
        y = F.relu(self.N9(self.L9(y)))
        y = F.relu(self.N10(self.L10(y)))
        y = F.relu(self.N11(self.L11(y)))
        return self.output(self.L12(self.D12(y)))

def create_balanced_sampler(dataset):
    def make_weights_for_balanced_classes(dataset, n_classes):
        count = [0] * n_classes
        for _, target in dataset:
            count[target] += 1
        weight_per_class = [0.] * n_classes
        N = float(sum(count))
        for i in range(n_classes):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(dataset)
        for idx, (_, target) in enumerate(dataset):
            weight[idx] = weight_per_class[target]
        return weight

    n_classes = np.unique(dataset.targets)
    weights = make_weights_for_balanced_classes(dataset, len(n_classes))
    return torch.utils.data.WeightedRandomSampler(weights, len(weights))

def train_eval(model, train_set, test_set, epochs=10, lr=0.01, mom=0.9, bs=32, show=True):
    # preparations
    model.to(DEVICE)
    train_loader = DataLoader(train_set, batch_size=bs, sampler=create_balanced_sampler(train_set))
    test_loader = DataLoader(test_set, batch_size=bs, sampler=create_balanced_sampler(test_set))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=mom)

    test_accuracy_display = []
    saved_states = []
    best_accuracy = 0
    best_epoch = 0

    # training
    if show:
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

            all_predictions, all_targets = [], []
            for signals_test, targets_test in test_loader:
                signals_test = signals_test.to(DEVICE)
                targets_test = targets_test.to(DEVICE)
                with torch.no_grad():
                    predictions = torch.argmax(model(signals_test), dim=1)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets_test.cpu().numpy())
            cmatrix = confusion_matrix(np.concatenate(all_predictions), np.concatenate(all_targets))
            accuracy = np.trace(cmatrix) / np.sum(cmatrix)
            test_accuracy_display.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = i_epoch
                torch.save({
                    'epoch': i_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'best_nn_layout.pth')

        if show:
            print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
                i_epoch+1, epochs, np.mean(train_losses), time.time()-start_time))

    if show:
        plt.figure(figsize=(10, 8))
        plt.plot(test_accuracy_display, label = "test accuracy")
        # plt.plot(train_loss_display, label = "train loss")
        plt.xlabel("Epoch")
        plt.ylabel('Accuracy')
        plt.title(f'Confusion Matrix for {model.__class__.__name__}')
        plt.show()

    if show:
        print(f"\n=== Evaluating {model.__class__.__name__} ===")

    checkpoint = torch.load('best_nn_layout.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # evaluation
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
    global_accuracy = np.trace(cmatrix)/np.sum(cmatrix)
    individual_accuracies = np.diag(cmatrix)/np.sum(cmatrix, axis=0)

    # results
    if show:
        print(f"Global accuracy: {global_accuracy:.4f}\nIndividual accuracies in descending order:")
        with np.printoptions(precision=4):
            print(np.sort(individual_accuracies)[::-1].reshape(-1, 7).T)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cmatrix, annot=True, fmt='d')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix for {model.__class__.__name__}')
        plt.show()

    return cmatrix, global_accuracy, individual_accuracies

def mean_train_eval(model_name, train_set, test_set, epochs=10, lr=0.01, mom=0.9, bs=32, show=True, n=10):
    mcm, gas, mia = np.zeros((35,35)), [], np.zeros(35)
    for i in range(1, n+1):
        cm, ga, ia = train_eval(model_name(), train_set, test_set, epochs=epochs, lr=lr, mom=mom, bs=bs, show=False)
        mcm += cm
        gas.append(ga)
        mia += ia
        print(f"Round {i} finished.")
    mcm = (mcm/n).astype(int)
    mga = np.mean(gas)
    mia /= n

    if show:
        print(f"\n=== Results of {n} rounds of train_eval for model {model_name.__name__}")
        print(f"Mean global accuracy: {mga:.4f} ± {np.std(gas):.4f}\nMean individual accuracies in descending order:")
        with np.printoptions(precision=4):
            print(np.sort(mia)[::-1].reshape(-1, 7).T)
        plt.figure(figsize=(10, 8))
        sns.heatmap(mcm, annot=True, fmt='d')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix for {model_name.__name__}')
        plt.show()

    return mcm, mga, mia

if __name__ == "__main__":
    train_set = PPGDataset("data/train8_reformat.xlsx")
    test_set = PPGDataset("data/test8_reformat.xlsx")

    train_set_segments = PPGDatasetSegments("data/train8_reformat.xlsx")
    test_set_segments = PPGDatasetSegments("data/test8_reformat.xlsx")

    # train_eval(PPGNetv1(), train_set, test_set, epochs=1000, bs=300, lr=0.1)
    # mean_train_eval(PPGNetv1, train_set, test_set, epochs=1000, bs=300, lr=0.1, n=10)

    # train_eval(PPGNetv2(), train_set, test_set, epochs=2500, bs=300, lr=0.01)
    train_eval(PPGNetv2(), train_set_segments, test_set_segments, epochs=20, bs=300, lr=0.01)
    # mean_train_eval(PPGNetv2, train_set, test_set, epochs=2500, bs=300, lr=0.01, n=10)

    # train_eval(PPGNetv3(), train_set, test_set, epochs=2500, bs=300, lr=0.01)
    # train_eval(PPGNetv3(), train_set_segments, test_set_segments, epochs=2500, bs=300, lr=0.01)
    # mean_train_eval(PPGNetv3, train_set, test_set, epochs=3000, bs=300, lr=0.01, n=10)
