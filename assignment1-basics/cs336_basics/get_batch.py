import numpy.typing as npt
import torch

# dataset: npt.NDArray, batch_size: int, context_length: int, device: str
# ) -> tuple[torch.Tensor, torch.Tensor]:
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(dataset) - context_length, (batch_size,))
    # The first tuple item is the sampled input sequences, and the second tuple item is the corresponding
    # language modeling labels.
    x = torch.stack([torch.tensor(dataset[i:i+context_length], device=device) for i in ix])
    y = torch.stack([torch.tensor(dataset[i+1:i+context_length+1], device=device) for i in ix])
    return x, y