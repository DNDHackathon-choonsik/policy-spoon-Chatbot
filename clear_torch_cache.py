import torch

def print_gpu_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3} GB")

# Example usage
print_gpu_memory()
