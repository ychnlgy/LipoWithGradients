import torch, math

from . import chebyshev, LagrangeBasis

import matplotlib
matplotlib.use("agg")

EPS = 1e-16

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
        self._axes = None

    def forward(self, X):
        B = self.basis(X)
        B = B.view(-1, self.d, self.n)
        L = (self.weight * B).sum(dim=-1)
        return L.view(X.size())

    def reset_parameters(self):
        self.weight.data.zero_()

    def visualize_relu(self, k, title, figsize):
        fig, self._axes = matplotlib.pyplot.subplots(
            nrows=1, ncols=k, sharex=True, sharey=True, figsize=figsize
        )
        axes = self._axes
        model = torch.nn.ReLU()
        n = 1000
        x = torch.linspace(-1, 1, n)
        y = model(x).cpu().numpy()
        x = x.cpu().numpy()
        for i in range(k):
            axes[i].plot(x, y)
            axes[i].set_xlim([-1, 1])
            axes[i].set_xlabel("$x_%d$" % i)
        axes[k//2].set_title(title)
        fname = "%s.png" % title
        matplotlib.pyplot.savefig(fname, bbox_inches="tight")
        print("Saved ReLU activations to %s" % fname)

    def visualize(self, mainplot, k, title, figsize):
        axes = mainplot[1,:]
        device = self.weight.device
        with torch.no_grad():
            #if self._axes is None:
            
            n = 1000
            for i in range(k):
                v = torch.linspace(-1, 1, n)
                X = torch.zeros(n, self.d)
                X[:,i] = v
                Xh = self.forward(X.to(device))

                plot = axes[i]
                plot.set_xlim([-1, 1])
                plot.plot(v.cpu().numpy(), Xh[:,i].cpu().numpy(), label="Interpolated polynomial activation")

                plot.plot(self.basis.nodes.cpu().numpy(), self.weight[0,i].clone().detach().cpu().numpy(), "x", label="Learned Chebyshev node")

                plot.axvline(x=self.basis.nodes[0].numpy(), linestyle=":", label="Chebyshev x-position")
                for node in self.basis.nodes[1:]:
                    plot.axvline(x=node.numpy(), linestyle=":")
            
                plot.set_xlabel("$x_%d$" % i)

        axes[k//2].legend(bbox_to_anchor=[1.1, -0.1])
        axes[k//2].set_ylabel("Polynomial output")
        mainplot[0,k//2].set_title(title)

        fname = "%s.png" % title
        matplotlib.pyplot.savefig(fname, bbox_inches="tight")
        print("Saved polynomial activations to %s" % fname)
        for i in range(2):
            for j in range(k):
                axes[i,j].cla()
