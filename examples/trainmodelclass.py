import torch
import torch.nn as nn
from torch.optim import Optimizer
from datasetattr import DatasetAttr


class TrainModelClass:
    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_fn: nn.Module) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def run_epoch(self, dataset: DatasetAttr, train: bool) -> float:
        total_loss = 0
        for data_point in dataset:
            x, y = data_point

            with torch.set_grad_enabled(train):
                out = self.model(x)
                loss = self.loss_fn(out, y)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / dataset.__len__()

    def train(self, epochs: int, train_data: DatasetAttr, test_data: DatasetAttr) -> None:
        for epoch_num in range(epochs):
            train_loss = self.run_epoch(train_data, True)
            test_loss = self.run_epoch(test_data, False)

            print("Epoch: {0}; Train Loss: {1:.3f}; Test Loss: {2:.3f}".format(epoch_num, train_loss, test_loss))

            train_data.on_epoch_end()
            test_data.on_epoch_end()
