import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.loss_fn = nn.CrossEntropyLoss()
        # custom cross entropy implementation
        self.loss_fn = lambda logits, targets: -torch.sum(torch.log_softmax(logits, dim=-1)[range(logits.size(0)), targets]) / logits.size(0)


    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits shape: (batch_size, sequence_length, vocab_size)
        # targets shape: (batch_size, sequence_length)
        # batch_size, seq_len, vocab_size = logits.shape
        # logits = logits.view(batch_size * seq_len, vocab_size)
        # targets = targets.view(batch_size * seq_len)

        # logits shape: (batch_size, vocab_size), targets shape: (batch_size)
        # subtract the largest logit for numerical stability
        logits = logits - logits.max(dim=-1, keepdim=True).values

        # perplexity
        # perplexity = torch.exp(self.loss_fn(logits, targets))
        return self.loss_fn(logits, targets)