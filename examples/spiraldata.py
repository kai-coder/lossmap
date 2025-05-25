import numpy as np
from numpy.typing import NDArray
from datasetattr import DatasetAttr
import torch


def generate_spiral_data(num: int, noise: float) -> tuple[NDArray[np.float_], NDArray[np.bool_]]:
    radii = 3 * np.pi * np.random.uniform(-1, 1, num)
    theta = np.abs(radii)

    noise_vector = noise * np.random.uniform(-2, 2, num)
    noisy_radii = radii + noise_vector

    x = noisy_radii * np.cos(theta)
    y = noisy_radii * np.sin(theta)

    coordinates = np.stack((x, y), axis=1)

    value = np.greater(radii, 0)

    return coordinates, value


class SpiralDataset(DatasetAttr):
    def __init__(self, num: int, noise: float, batch_size: int, shuffle: bool) -> None:
        super().__init__(batch_size, shuffle)
        self.x, self.y = generate_spiral_data(num, noise)
        self.x /= self.x.max()

        self.on_epoch_end(True)

    def __len__(self) -> int:
        return len(self.x) // self.batch_size

    def on_epoch_end(self, must_shuffle: bool=False) -> None:
        index_arr = np.arange(len(self.x))

        if self.shuffle or must_shuffle:
            np.random.shuffle(index_arr)

        self.x = self.x[index_arr]
        self.y = self.y[index_arr]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx >= self.__len__():
            raise IndexError()

        next_idx = (idx + 1) * self.batch_size

        x = torch.tensor(self.x[idx:next_idx],  dtype=torch.float32)
        y = torch.tensor(self.y[idx:next_idx, None],  dtype=torch.float32)

        return x, y
