import torch, math

from . import chebyshev, LagrangeBasis

class Activation(torch.nn.Module):

    def __init__(self, input_size, n_degree):
        super().__init__()
        self.d = input_size
        self.n = n_degree + 1
        self.basis = LagrangeBasis.create(
            chebyshev.get_nodes(n_degree+1)
        )
        self.weight = torch.nn.Parameter(
            torch.zeros(1, self.d, self.n)
        )

    def forward(self, X):
        B = self.basis(X)
        B = B.view(-1, self.d, self.n)
        L = (self.weight * B).sum(dim=-1)
        return L.view(X.size())

    def reset_parameters(self):
        self.weight.data.zero_()
