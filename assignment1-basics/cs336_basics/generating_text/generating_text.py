import torch
import torch.nn as nn

import os
import typing

from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.attention.softmax import softmax

def generating_next_token_id(
    input_seq: torch.Tensor,
    model: nn.Module,

    # Sampling
    top_p_threshold: float,

    # Softmax Temperature
    temperature: float = 0,
):  

    # Forward Pass
    y_logits = model(input_seq) # batch_size = 1, sequence_length, vocab_size

    # Select Batch -> Sequence -> Logit for Next Token and Sort
    logit, logit_indice = torch.sort(y_logits[0][-1], dim = -1, descending = True) # sequence_length, vocab_size

    # Turn it into Normalized Probability with Temperature
    if temperature == 0:
        return logit_indice[0] # Argmax        
    else:
        prob = softmax(logit / temperature, dim = -1)

    # Calculate Cumulative Prob
    prob_cum = torch.cumsum(prob, dim = -1)

    # Mask
    mask = prob_cum - prob < top_p_threshold

    # Truncated Result
    prob_truncated = prob[mask]

    # Normalize Prob
    prob_truncated_normalized = prob_truncated / torch.sum(prob_truncated, dim = -1)

    # Select Max P
    sample_idx = torch.multinomial(prob_truncated_normalized, num_samples = 1)

    # Next Token ID
    next_tok_id = logit_indice[sample_idx[0]]

    return next_tok_id


def generating_text(
    input_seq: torch.Tensor,
    model_state_dict_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    
    # EOS Token ID
    eos_token_id: int,

    # Transformer
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,

    # Sampling
    top_p_threshold: float,

    # Max Number of Generated Tokens
    max_num_generated_tok: int,

    # Softmax Temperature
    temperature: float = 0,

    device: str | torch.device | None = None,
):
    # Length of Input
    current_input = input_seq
    current_input = current_input.to(device)

    # Init Model    
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model = model.to(device)

    # Load State Dict
    checkpoint = torch.load(model_state_dict_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Put Model in Eval Mode
    model.eval()

    # Turn on Inference Mode to Disable Gradient Tracking, etc.
    with torch.inference_mode():
        
        # Max Number of Generated Tokens
        for _ in range(max_num_generated_tok):
            

            # Get Next Token ID
            next_tok_id = generating_next_token_id(current_input[:, -min(current_input.shape[-1], context_length):], model, top_p_threshold, temperature)

            # Concate into Input
            current_input = torch.cat([current_input, next_tok_id.reshape(1,1)], dim = -1)

            # EOS
            if next_tok_id == eos_token_id:
                break
    
    return current_input