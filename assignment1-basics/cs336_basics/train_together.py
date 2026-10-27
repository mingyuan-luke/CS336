import os
import torch
import time
import wandb
import numpy as np
from cs336_basics.get_batch import get_batch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import CrossEntropyLoss
from cs336_basics.gradient_clipping import gradient_clipping

# Ability to configure and control the various model and optimizer hyperparameters
EXP_NAME = "first_model"

# Memory-efficient loading of large training and validation datasets with np.memmap
TRAIN_DATA = "results/ts_train.npy"
VAL_DATA = "results/ts_valid.npy"

# Checkpointing to save and resume training progress, Serializing checkpoints to a user-provided path.
CHECKPOINT_PATH = "results/checkpoints"
# Periodically logging training and validation performance
LOG_INTERVAL = 100
# Number of training iterations
MAX_ITER = 10000
# save ckpt periodically during training
CKPT_INTERVAL = 1000
# Interval for performing validation during training
VAL_INTERVAL = 500  # Interval for validation during training
# Batch size for training
BATCH_SIZE = 32
# Number of epochs for training
EPOCHS = 10
# Logging configuration
LOG_FILE = "results/training.log"
# device for training (e.g., "cpu" or "cuda" or "mps")
DEVICE = "mps"  # Change to "cuda" or "mps" if available

# model hyperparameters
VOCAB_SIZE = 10000  # Example vocab size, replace with actual
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10000.0

# optimizer
LR = 1e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01

# initialize model and optimizer
# vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float
MODEL = TransformerLM(
    vocab_size=VOCAB_SIZE,  # Example vocab size, replace with actual
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    rope_theta=ROPE_THETA
)
MODEL.to(DEVICE)
OPTIMIZER = AdamW(MODEL.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
LOSS_CE = CrossEntropyLoss()
# 文件头不能被当作 token, dtype uint16, 内存映射模式
get_train_data = np.load(TRAIN_DATA, mmap_mode='r')
# print(f"Train data shape: {get_train_data.shape}, dtype: {get_train_data.dtype}, first ten: {get_train_data[:10]}")
# print(f"Max value in train data: {get_train_data.max()}")
get_val_data = np.load(VAL_DATA, mmap_mode='r')
# print(f"Validation data shape: {get_val_data.shape}, dtype: {get_val_data.dtype}, first ten: {get_val_data[:10]}")
# print(f"Max value in validation data: {get_val_data.max()}")

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="lukemillerforge-tianjin-university",
    # Set the wandb project where this run will be logged.
    project="cs336",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": LR,
        "architecture": "TransformerLM",
        "dataset": "TinyStoriesV2",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "d_ff": D_FF,
        "rope_theta": ROPE_THETA,
    },
)

# clear train log
if os.path.exists(LOG_FILE):
    open(LOG_FILE, "w").close()

# train loop
start_time = time.time()
for epoch in range(EPOCHS):
    for iter in range(MAX_ITER):
        # get batches for training using get_batch function
        batch = get_batch(get_train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        # conversion for inputs and targets，torch.long
        inputs, targets = batch # target shape: (batch_size, context_length)
        inputs = inputs.to(torch.long)
        targets = targets.to(torch.long)
            
        # Forward pass, loss computation, backward pass, and optimizer step would go here
        outputs = MODEL(inputs) # shape: (batch_size, context_length, vocab_size)
        # outputs dimention need to adjust
        outputs = outputs.view(-1, outputs.size(-1)) # shape: (batch_size * context_length, vocab_size)
        targets = targets.view(-1) # shape: (batch_size * context_length)
        loss = LOSS_CE(outputs, targets)
        loss.backward()
        # Gradient clipping
        gradient_clipping(MODEL.parameters(), max_l2_norm=1.0)
        # Assuming an optimizer is defined, e.g., OPTIMIZER.step() and OPTIMIZER.zero_grad() would be called here
        if 'OPTIMIZER' in globals():
            OPTIMIZER.step()
            OPTIMIZER.zero_grad()

        # record training loss, print time elapsed
        if iter % LOG_INTERVAL == 0:
            print(f"Epoch [{epoch}/{EPOCHS}], Iter [{iter}/{MAX_ITER}], Loss: {loss.item()}, Time: {time.time() - start_time}")
            with open(LOG_FILE, "a") as f:
                f.write(f"Epoch [{epoch}/{EPOCHS}], Iter [{iter}/{MAX_ITER}], Loss: {loss.item()}, Time: {time.time() - start_time}\n")
            run.log({"train_loss": loss.item(), "iter": iter})
            start_time = time.time()

        # 将检查点序列化到用户提供的路径
        if iter % CKPT_INTERVAL == 0 or iter == MAX_ITER - 1:
            os.makedirs(CHECKPOINT_PATH, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'iter': iter,
                'model_state_dict': MODEL.state_dict() if hasattr(MODEL, 'state_dict') else None,
                'optimizer_state_dict': OPTIMIZER.state_dict() if 'OPTIMIZER' in globals() else None,
            }, f"{CHECKPOINT_PATH}/checkpoint_epoch_{epoch}_iter_{iter}.pth")

        # 验证性能（例如，到控制台和/或像 Weights and Biases 这样的外部服务
        if iter % VAL_INTERVAL == 0 or iter == MAX_ITER - 1:
            val_batch = get_batch(get_val_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
            val_inputs, val_targets = val_batch
            val_inputs = val_inputs.to(torch.long)
            val_targets = val_targets.to(torch.long)
            val_outputs = MODEL(val_inputs)
            val_outputs = val_outputs.view(-1, val_outputs.size(-1))
            val_targets = val_targets.view(-1)
            val_loss = LOSS_CE(val_outputs, val_targets)
            print(f"Epoch [{epoch}/{EPOCHS}], Iter [{iter}/{MAX_ITER}], Validation Loss: {val_loss.item()}")
            with open(LOG_FILE, "a") as f:
                f.write(f"Epoch [{epoch}/{EPOCHS}], Iter [{iter}/{MAX_ITER}], Validation Loss: {val_loss.item()}\n")
            # record iteration and validation loss to wandb
            run.log({"val_loss": val_loss.item(), "iter": iter})

# Finish the run and upload any remaining data.
run.finish()


# usage example:
# uv run python cs336_basics/train_together.py