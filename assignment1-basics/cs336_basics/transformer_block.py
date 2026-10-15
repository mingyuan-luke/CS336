import torch
import torch.nn as nn

from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.rmsnorm1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True)
        
        self.rmsnorm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_ff, d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # Apply first RMSNorm
        x_norm = self.rmsnorm1(x)

        # use all token positions by default
        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        
        # Apply multi-head self-attention with RoPE
        attn_out = self.attn(x_norm, mask=None, theta=self.theta, max_seq_len=self.max_seq_len, token_positions=token_positions)

        # Residual connection
        x = x + attn_out

        # Apply second RMSNorm
        x_norm = self.rmsnorm2(x)

        # Apply feed-forward network
        ff_out = self.ff(x_norm)

        # Residual connection
        x = x + ff_out

        return x