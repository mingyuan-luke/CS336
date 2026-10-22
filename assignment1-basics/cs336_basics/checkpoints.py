import torch
from cs336_basics.adamw import AdamW

# optimizer uses AdamW, so we save its param_groups and state explicitly
def save_checkpoint(model, optimizer, iteration, out):
    # save optimizer param_groups and state as part of the checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']