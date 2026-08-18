import torch
import numpy.typing as npt

import os
import typing

from collections.abc import Callable

from cs336_basics.training_loop.get_batch import get_batch
from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.training.cross_entropy import cross_entropy
from cs336_basics.training.adamw import AdamW
from cs336_basics.training.gradient_clipping import clip_grad_norm_
from cs336_basics.training_loop.checking import save_checkpoint, load_checkpoint

def training(

    epochs: int,
    path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], # Path to Save and Load Checkpoint
    
    # Get Batch
    train_dataset: npt.NDArray,
    val_dataset: npt.NDArray,
    batch_size: int,
    context_length: int,

    # Transformer LM
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,

    # Optimizer
    max_learning_rate: float | None,
    min_learning_rate: float | None,
    warmup_iters: int | None,
    cosine_cycle_iters: int | None,
    lr = 1e-3,
    lr_schedule: Callable[[int, float, float, int, int], float] | None = None,
    betas = (0.9, 0.999),
    eps = 1e-8,
    weight_decay: float = 0.01,

    # Gradient Clip
    max_l2_norm: float = 1.0,
    
    # Training Check
    new_training: bool = True,

    device: str | torch.device | None = None,
    # dtype: torch.dtype | None = None,
):

    assert batch_size > 0, "Batch Size Should Greater Than 0"
    assert context_length > 0, "Context Length Should Greater Than 0"
    assert len(train_dataset) > context_length, "Train Dataset Length Should Greater Than Context Length"
    assert len(val_dataset) > context_length, "Val Dataset Length Should Greater Than Context Length"

    # Init Model
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    
    # Get Model Weight for Optimizer
    # Use list() for Later Use by Grad Clip Because .parameters() Return Iterater (Can Only Iterate Once)
    params = list(model.parameters())

    # Init Optimizer
    optimizer = AdamW(params, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
    
    # Calculate Step Required to Complete Training on Number of Token that is Approximately Equal Dataset Size
    train_steps = max(
        1,
        len(train_dataset) // (batch_size * context_length),
    )

    # Load Checkpoint if Not New Training
    if not new_training:
        last_epoch = load_checkpoint(path, model, optimizer)
    else:
        last_epoch = -1

    # Store Loss
    Loss_Training = []
    Loss_Validation = []

    # Start Training from Last Epoch
    for epoch in range(last_epoch + 1, epochs):
        
        # Put Model in Training Mode
        model.train()

        # Define Train Loss
        train_loss = 0

        for step in range(train_steps):
            # Update lr if Using LR Schedule
            if lr_schedule is not None:
                new_lr = lr_schedule(step + epoch * train_steps, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
                for group in optimizer.param_groups:
                    group["lr"] = new_lr

            # Zero the Gradients
            optimizer.zero_grad()

            # Load New Batch Data
            x, y = get_batch(train_dataset, batch_size, context_length, device)

            # Forward Pass
            y_logits = model(x)

            # Calculate the Loss
            loss = cross_entropy(y_logits, y)
            train_loss += loss.item()

            # Backpropagation on Loss
            loss.backward()

            # Gradient Clipping
            clip_grad_norm_(params, max_l2_norm)

            # Step the Optimizer
            optimizer.step()

        # Store Checkpoint For Every Epoch
        save_checkpoint(model, optimizer, epoch, path)

        # Store Loss After Each Epoch
        Loss_Training.append(train_loss / train_steps)

        # Put Model in Eval Mode
        model.eval()

        # Turn on Inference Mode to Disable Gradient Tracking, etc.
        with torch.inference_mode():
            
            # Calculate Step Required to Complete Validation on Number of Token that is Approximately Equal Dataset Size
            val_steps = max(
                1,
                len(val_dataset) // (batch_size * context_length),
            )

            val_loss = 0

            for i in range(val_steps):
                # Load New Batch Data
                x, y = get_batch(val_dataset, batch_size, context_length, device)

                # Forward Pass
                y_logits = model(x)

                # Calculate the Loss
                loss = cross_entropy(y_logits, y)
                val_loss += loss.item()
                
            # Store Validation Loss for Every Epoch
            Loss_Validation.append(val_loss / val_steps)

    return Loss_Training, Loss_Validation

        