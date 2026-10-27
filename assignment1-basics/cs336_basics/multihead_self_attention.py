import torch
import torch.nn as nn
from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
from cs336_basics.rope import *

class MultiheadSelfAttention(nn.Module):
    # need support rope if using rotary positional embeddings
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.sdpa = ScaledDotProductAttention(d_k=self.d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, theta: float | None = None, max_seq_len: int | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # shape (batch_size, num_heads, seq_len, d_k)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # apply rope if use_rope is True
        if self.use_rope and theta is not None and max_seq_len is not None and token_positions is not None:
            Q, K = self.apply_rope(Q, K, theta=theta, max_seq_len=max_seq_len, token_positions=token_positions)
        # need mask even is none, can not see future tokens
        if mask is None:
            mask = torch.ones(batch_size, seq_len, seq_len, device=x.device, dtype=torch.bool).triu(diagonal=1) == 0 # shape (batch_size, seq_len, seq_len)
            # mask shape must equal to scores
            if mask.shape != (batch_size, self.num_heads, seq_len, seq_len):
                mask = mask.unsqueeze(1)

        attn_output = self.sdpa(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(attn_output)

    # implement rope
    def apply_rope(self, Q: torch.Tensor, K: torch.Tensor, theta: float, max_seq_len: int, token_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Q_rope = run_rope(d_k=self.d_k, theta=theta, max_seq_len=max_seq_len, in_query_or_key=Q, token_positions=token_positions)
        K_rope = run_rope(d_k=self.d_k, theta=theta, max_seq_len=max_seq_len, in_query_or_key=K, token_positions=token_positions)
        return Q_rope, K_rope
    