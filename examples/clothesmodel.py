import torch
import lossmap
import torch.nn as nn
from spiraldata import ClothesDataset
import matplotlib.pyplot as plt
from trainmodelclass import TrainModelClass
from functools import partial
import torchvision
import numpy as np
import os

print(os.getenv("MNIST_PATH"))
trainData = torchvision.datasets.FashionMNIST(os.environ.get("MNIST_PATH"), True)
trainX = np.array([np.array(i[0]) for i in trainData])[:20000]
trainy = np.array([i[1] for i in trainData])[:20000]

testData = torchvision.datasets.FashionMNIST(os.environ.get("MNIST_PATH"), False)
testX = np.array([np.array(i[0]) for i in testData])[:3000]
testy = np.array([i[1] for i in testData])[:3000]

trainData = ClothesDataset(trainX, trainy, 256, True)
testData = ClothesDataset(testX, testy, 256, False)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flat = nn.Flatten()
        self.conv1 = nn.Linear(28**2, 64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flat(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        return x


model = Model()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
train_model_class = TrainModelClass(optimizer, loss_fn)

train_model_class.train(10, trainData, testData, model)

loss_map = lossmap.LossMap(model)
x, y, loss = loss_map.get_loss_landscape(-1, 1, 100, partial(train_model_class.run_epoch, trainData, False))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_wireframe(x, y, loss, color='C0')
plt.show()

