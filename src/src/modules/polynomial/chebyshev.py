import torch, math

def get_nodes(n, a, b):
    x = chebyshev_node(torch.arange(1, n+1).float(), n)
    return 0.5*(a+b)+0.5*(b-a)*x

# === PRIVATE ===

def chebyshev_node(k, n):
    return torch.cos((2*k-1)*math.pi/(2*n))
