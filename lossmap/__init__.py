import torch.nn as nn


class LossMap:
    def __init__(self, model: nn.Module):
        self.model = model
