import torch.nn as nn
import torch
import numpy as np
from collections.abc import Callable
from numpy.typing import NDArray


def add_param_vectors(param_vector: tuple[list[torch.Tensor], ...]) -> list[torch.Tensor]:
    added_params = []
    for params in zip(*param_vector):
        stacked_params = torch.stack(params)
        added_params.append(torch.sum(stacked_params, dim=0))

    return added_params


class LossMap:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.model_params = self.get_params()
        self.device = device

    def get_params(self) -> list[torch.Tensor]:
        model_params = []
        with torch.no_grad():
            for params in self.model.parameters():
                model_params.append(params.data.clone())
        return model_params

    def get_unit_param_vector(self) -> list[torch.Tensor]:
        random_vector = []
        total_length = 0

        for params in self.model_params:
            random_params = torch.rand(params.shape)
            random_params = random_params * 2 - 1

            total_length += torch.sum(random_params ** 2)
            random_vector.append(random_params)

        total_length = np.sqrt(total_length)

        for params in random_vector:
            params /= total_length

        return random_vector

    def update_params(self, param_vector: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for params, vector in zip(self.model.parameters(), param_vector):
                params += vector

    def get_loss_landscape(self, min_explore: float, max_explore: float, data_num: float,
                           eval_fn: Callable[[nn.Module], float]) \
            -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_],]:
        vector1 = self.get_unit_param_vector()
        vector2 = self.get_unit_param_vector()

        total_steps = int(np.floor(np.sqrt(data_num)))

        steps = np.linspace(min_explore, max_explore, total_steps)

        x, y = np.meshgrid(steps, steps)
        coordinates = np.stack((x.ravel(), y.ravel()), axis=1)

        loss_map = np.zeros(len(coordinates))

        for step_num in range(len(coordinates)):
            weight1, weight2 = coordinates[step_num]

            weighted_vector1 = [weight1 * param for param in vector1]
            weighted_vector2 = [weight2 * param for param in vector2]

            added_vectors = add_param_vectors((weighted_vector1, weighted_vector2))

            with torch.no_grad():
                for (params, idx) in zip(self.model.parameters(), range(len(self.model_params))):
                    new_params = self.model_params[idx] + added_vectors[idx]
                    params.copy_(new_params.to(self.device))

                loss_map[step_num] = eval_fn(self.model)

        return x, y, loss_map.reshape(x.shape)
