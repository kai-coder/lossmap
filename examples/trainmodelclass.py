import torch
import torch.nn as nn
from torch.optim import Optimizer
from datasetattr import DatasetAttr


class TrainModelClass:
    def __init__(self, optimizer: Optimizer, loss_fn: nn.Module) -> None:
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def run_epoch(self, dataset: DatasetAttr, train: bool, model: nn.Module) -> float:
        total_loss = 0
        for data_point in dataset:
            x, y = data_point

            if train:
                model.train()
            else:
                model.eval()

            with torch.set_grad_enabled(train):
                out = model(x)
                loss = self.loss_fn(out, y)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / dataset.__len__()

    def train(self, epochs: int, train_data: DatasetAttr, test_data: DatasetAttr, model: nn.Module) -> None:
        for epoch_num in range(epochs):
            train_loss = self.run_epoch(train_data, True, model)
            test_loss = self.run_epoch(test_data, False, model)

            print("Epoch: {0}; Train Loss: {1:.3f}; Test Loss: {2:.3f}".format(epoch_num, train_loss, test_loss))

            train_data.on_epoch_end()
            test_data.on_epoch_end()


class FancyTrainModelClass(TrainModelClass):
    def __init__(self, optimizer: Optimizer, loss_fn: nn.Module, scheduler: nn.Module) -> None:
        super().__init__(optimizer, loss_fn)
        self.scaler = torch.GradScaler("cuda")
        self.scheduler = scheduler

    def run_epoch(self, dataset: DatasetAttr, train: bool, model: nn.Module) -> float:
        total_loss = 0
        for data_point in dataset:
            x, y = data_point

            if train:
                model.train()
            else:
                model.eval()

            with torch.set_grad_enabled(train):
                out = model(x)
                loss = self.loss_fn(out, y)

            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item()

        return total_loss / dataset.__len__()

    def train(self, epochs: int, train_data: DatasetAttr, test_data: DatasetAttr, model: nn.Module)\
            -> tuple[list[float], list[float]]:
        train_loss_arr = []
        test_loss_arr = []
        for epoch_num in range(epochs):
            train_loss = self.run_epoch(train_data, True, model)
            test_loss = self.run_epoch(test_data, False, model)

            print("Epoch: {0}; Learning Rate: {1:.5f}".format(epoch_num, self.scheduler.get_last_lr()))
            print("Train Loss: {0:.3f}; Test Loss: {1:.3f}".format(train_loss, test_loss))
            print()

            train_loss_arr.append(train_loss)
            test_loss_arr.append(test_loss)

            self.scheduler.step()
            train_data.on_epoch_end()
            test_data.on_epoch_end()

        return train_loss_arr, test_loss_arr
