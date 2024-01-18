#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
A Simple example of Sequence Prediction adopting Data Augmentation Technique.
Tensorflow and Keras Packages Based.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# from torchvision.transforms import RandomOrder
import random


def RandomOrderTransform(transforms):  # transforms shape: (30,10,5)
        # if not isinstance(x, (tuple, list)):
        #     x = list(x)
        x = []
        # assert isinstance(x, list), "Input must be a list or tuple"
        order = list(range(len(transforms)))
        random.shuffle(order)  # shuffle的index for first dimension
        for i in order:
            x.append(transforms[i])
        # if isinstance(x, list):
        #     x = tuple(x)
        return x


# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):  # 5, 16, 1
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = torch.Tensor(x)
        x = torch.stack(x, dim=0)
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        # h0 = torch.zeros(1, x.size(dim=1), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(1, x.size(dim=1), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Define the contrastive learning with data augmentation
class ContrastiveLearning(nn.Module):
    def __init__(self):
        super(ContrastiveLearning, self).__init__()
        # self.rand_order = RandomOrder()
        # self.rand_order = RandomOrderTransform()

    def forward(self, x):
        # x1 = self.rand_order(x)
        # x2 = self.rand_order(x)
        x1 = RandomOrderTransform(x)
        x2 = RandomOrderTransform(x)
        return x1, x2


# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs = data.float().to(device)
        optimizer.zero_grad()
        x1, x2 = contrastive_learning(inputs)
        # 这里是两个augmentated data都要input到model一次
        outputs1 = model(x1)
        outputs2 = model(x2)
        loss = criterion(outputs1, outputs2)
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        # print("loss: ", loss)
        running_loss += loss[0].item()
    return running_loss / len(train_loader)


if __name__ == '__main__':
    # Set up the data
    train_data = torch.randn(1000, 10, 5)
    train_dataset = MyDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Set up the model
    input_dim = 5
    hidden_dim = 16
    output_dim = 1
    model = LSTMModel(input_dim, hidden_dim, output_dim)
    contrastive_learning = ContrastiveLearning()

    # Set up the loss function and optimizer
    criterion = nn.CosineSimilarity(dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")


