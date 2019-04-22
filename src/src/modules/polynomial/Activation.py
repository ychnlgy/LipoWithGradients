import torch, math

from . import chebyshev, LagrangeBasis

import matplotlib
matplotlib.use("agg")

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

    def visualize(self, k, title, figsize):
        device = self.weight.device
        with torch.no_grad():
            fig, axes = matplotlib.pyplot.subplots(
                nrows=k, ncols=1, sharex=True, sharey=True, figsize=figsize
            )
            n = 1000
            for i in range(k):
                v = torch.linspace(-1, 1, n)
                X = torch.zeros(n, self.d)
                X[:,i] = v
                Xh = self.forward(X.to(device))

                plot = axes[i]
                plot.plot(v.cpu().numpy(), Xh[:,i].cpu().numpy(), label="Interpolated polynomial activation")

                plot.plot(self.basis.nodes.cpu().numpy(), self.weight[0,i].cpu().numpy(), "x", label="Learned Chebyshev node")

                for node in self.basis.nodes:
                    plot.axvline(x=node.numpy(), ":", label="Chebyshev x-position")
            
                plot.set_ylabel("$x_%d$" % i)
        axes[-1].legend()
        axes[0].set_title(title)

        fname = "%s.png" % title
        matplotlib.pyplot.savefig(fname)
        print("Saved polynomial activations to %s" % fname)
