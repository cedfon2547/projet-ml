import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    print("hello")
