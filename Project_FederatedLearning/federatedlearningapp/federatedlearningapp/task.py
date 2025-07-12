"""FederatedLearningApp: A Flower / PyTorch app using ImageFolder layout for CIFAR-10."""

import os
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DATA_DIR = "data/cifar10"   # should contain subfolders "train/" and "test/"
BATCH_SIZE = 32


class Net(nn.Module):
    """Simple CNN (adapted from the PyTorch 60-Minute Blitz)."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data(partition_id: int, num_partitions: int):
    """
    Load CIFAR-10 from Kaggle-downloaded folders and split IID across clients.
    Expects:
        data/cifar10/train/  <-- 10 class-named subfolders of PNGs
        data/cifar10/test/   <-- same structure
    Returns:
        trainloader, testloader for shard `partition_id`
    """
    # 1) Define transforms
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 2) Create ImageFolder datasets
    train_dataset = ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
    test_dataset  = ImageFolder(root=os.path.join(DATA_DIR, "test"),  transform=transform)

    # 3) Helper to shard indices
    def shard_indices(dataset_size: int):
        idx = list(range(dataset_size))
        random.seed(42)
        random.shuffle(idx)
        shard_sz = dataset_size // num_partitions
        start = partition_id * shard_sz
        # last shard gets any remainder
        end = dataset_size if partition_id == num_partitions - 1 else start + shard_sz
        return idx[start:end]

    train_inds = shard_indices(len(train_dataset))
    test_inds  = shard_indices(len(test_dataset))

    # 4) Wrap in Subset + DataLoader
    trainloader = DataLoader(
        Subset(train_dataset, train_inds),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    testloader = DataLoader(
        Subset(test_dataset, test_inds),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return trainloader, testloader


def train(net: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device):
    """
    Train the model on the local shard. Returns the average training loss.
    """
    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    total_batches = len(trainloader) * epochs

    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / total_batches


def test(net: nn.Module, testloader: DataLoader, device: torch.device):
    """
    Evaluate the model on the local test set. Returns (avg_loss, accuracy).
    """
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()

    loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return avg_loss, accuracy


def get_weights(net: nn.Module):
    """
    Extract model parameters as a list of NumPy arrays (for Flower).
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters):
    """
    Load model parameters from a list of NumPy arrays (for Flower).
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
