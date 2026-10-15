import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device('cpu')

        # RoPE frequency: for pair index j in [0, 1, ..., d/2-1], 其实这就是k从1到d/2 
        # frequency = theta^(-2j/d) where 2j corresponds to arange(0, d_k, 2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=self.device).float() / d_k)) # shape (d_k/2)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute the sinusoidal embeddings for all positions up to max_seq_len
        t = torch.arange(max_seq_len, device=self.device).float()
        freqs = torch.einsum("i , j -> i j", t, inv_freq) # shape (max_seq_len, d_k/2)

        # [f_0, f_0, f_1, f_1, ..., f_{d/2-1}, f_{d/2-1}]
        emb = torch.repeat_interleave(freqs, 2, dim=-1) # shape (max_seq_len, d_k)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x (..., sequence_length, d_k)
        cos = self.cos_cached[token_positions, :].unsqueeze(0) # shape (1, sequence_length, d_k)
        sin = self.sin_cached[token_positions, :].unsqueeze(0)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]  # even indices
    x2 = -x[..., 1::2]  # odd indices
    # concat x2[0], x1[0], x2[1], x1[1], ... along the last dimension
    return torch.stack((x2, x1), dim=-1).flatten(-2)

# apply rope
def run_rope(d_k: int, theta: float, max_seq_len: int, in_query_or_key: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    # in_query_or_key (..., sequence_length, d_k)
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    cos, sin = rope(in_query_or_key, token_positions)
    return (in_query_or_key * cos) + (rotate_half(in_query_or_key) * sin)


