import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # Define the Max Start Index for Sampling
    max_start_idx = len(dataset) - context_length - 1

    # Create List Start Index with Shape (Batch_Size, 1)
    start_indices = np.random.randint(low = 0, high = max_start_idx + 1, size = (batch_size, 1))

    # Create List with Context Length with Shape (1, Context_Length)
    context_offsets = np.arange(context_length).reshape((1, context_length))
    
    # Broadcasting to Form (Batch_Size, Context_Length)
    input_indices = start_indices + context_offsets
    target_indices = (start_indices + 1) + context_offsets

    # Output Seq from Dataset
    input_batch = dataset[input_indices]
    target_batch = dataset[target_indices]

    # Move to Device
    input_batch = torch.tensor(input_batch, device = device, dtype = torch.int64)
    target_batch = torch.tensor(target_batch, device = device, dtype = torch.int64)

    return (input_batch, target_batch)