import torch
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.softmax import Softmax

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta
            )
            for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)
        self.softmax = Softmax()

    def forward(self, in_indices: torch.IntTensor) -> torch.FloatTensor:
        x = self.token_embeddings(in_indices)
        token_positions = torch.arange(in_indices.size(1), device=in_indices.device).unsqueeze(0).expand(in_indices.size(0), -1)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.rmsnorm(x)
        x = self.output_linear(x) # shape: (batch_size, context_length, vocab_size)
        # x = self.softmax(x)
        return x