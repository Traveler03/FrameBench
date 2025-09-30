import time
import torch

def benchmark(func, num_repeats, *args, **kwargs):
    # warm up
    for _ in range(10):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_repeats):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    return elapsed / num_repeats * 1000  # return ms 