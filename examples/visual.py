import torch

import lossmap
import numpy as np
import torch.nn as nn
from spiraldata import SpiralDataset
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from trainmodelclass import TrainModelClass


trainData = SpiralDataset(1000, 0.5, 32, True)
testData = SpiralDataset(1000, 0.5, 32, False)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Linear(2, 64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        return x


model = Model()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss()
train_model_class = TrainModelClass(model, optimizer, loss_fn)

train_model_class.train(50, trainData, testData)
