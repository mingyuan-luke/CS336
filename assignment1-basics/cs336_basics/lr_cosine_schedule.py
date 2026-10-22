

def lr_cosine_schedule(step_t, lr_max, lr_min, warmup_steps, total_steps):
    """
    Compute the learning rate at a given step using cosine annealing schedule.

    Args:
        step_t (int): Current step.
        lr_max (float): Maximum learning rate.
        lr_min (float): Minimum learning rate.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps.

    Returns:
        float: Adjusted learning rate.
    """
    import math
    if step_t < warmup_steps:
        return lr_max * step_t / warmup_steps
    if step_t >= total_steps:
        return lr_min
    progress = (step_t - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))