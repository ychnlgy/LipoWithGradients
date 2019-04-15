import torch, math

def get_nodes(n):
    return chebyshev_node(torch.arange(1, n+1).float(), n)

# === PRIVATE ===

def chebyshev_node(k, n):
    return torch.cos((2*k-1)*math.pi/(2*n))
