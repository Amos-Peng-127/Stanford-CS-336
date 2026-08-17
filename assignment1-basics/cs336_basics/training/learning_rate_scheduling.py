import math

def learning_rate_scheduling(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters

    if it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + \
            0.5 * (1 + math.cos((it - warmup_iters) * \
                math.pi / (cosine_cycle_iters - warmup_iters))) * \
                    (max_learning_rate - min_learning_rate)
    
    if it > cosine_cycle_iters:
        return min_learning_rate
