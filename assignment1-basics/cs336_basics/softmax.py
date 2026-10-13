import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.softmax(x, dim=self.dim)
        # get dim in x, minus max value
        x_max = x.max(dim=self.dim, keepdim=True).values
        x_exp = torch.exp(x - x_max)
        return x_exp / x_exp.sum(dim=self.dim, keepdim=True)