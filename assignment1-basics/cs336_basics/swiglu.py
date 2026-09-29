import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    # d_model: int,
    # d_ff: int,
    # w1_weight: Float[Tensor, " d_ff d_model"],
    # w2_weight: Float[Tensor, " d_model d_ff"],
    # w3_weight: Float[Tensor, " d_ff d_model"],
    # in_features: Float[Tensor, " ... d_model"],
    def __init__(self, d_ff, d_model):
        super(SwiGLU, self).__init__()
        # w1, w2, and w3 are the weight matrices for the SwiGLU layer
        # no bias
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        # x1: (..., d_ff)
        x1 = self.w1(x)
        # use sigmoid for silu， x2: (..., d_ff)
        x2 = x1 * torch.sigmoid(x1)
        # x3: (..., d_ff)
        x3 = x2 * self.w3(x)
        return self.w2(x3)
