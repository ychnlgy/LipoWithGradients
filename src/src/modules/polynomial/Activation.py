import torch, math

from . import chebyshev, LagrangeBasis

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

EPS = 1e-16

class Activation(torch.nn.Module):

    def __init__(self, input_size, n_degree):
        super().__init__()
        b = self._calc_chebyshev_range(n_degree+1)
        a = -b
        self.d = input_size
        self.n = n_degree + 1
        self.basis = LagrangeBasis.create(
            chebyshev.get_nodes(n_degree+1, a, b)
        )
        self.weight = torch.nn.Parameter(
            torch.zeros(1, self.d, self.n)
        )
        self.r = (a, b)
        self._axes = None

    def forward(self, X):
        '''

        Input:
            X - torch Tensor of size (N, D, *), input features.

        Output:
            P - torch Tensor of size (N, D, *), polynomial output.

        '''
        B = self.basis(X)
        e = len(B.shape) - len(self.weight.shape)
        w = self.weight.view(1, self.d, *([1]*e), self.n)
        L = (w * B).sum(dim=-1)
        assert L.size() == X.size()
        return L

    def _calc_chebyshev_range(self, n):
        return 1.0/math.cos(math.pi/(2*n))

    def reset_parameters(self):
        self.weight.data.zero_()

    def visualize(self, k, title, figsize):
        if self._axes is None:
            _, self._axes = pyplot.subplots(nrows=1, ncols=k, sharex=True, sharey="row", figsize=figsize)
        mainplot = self._axes
        axes = mainplot
        top = mainplot
        
        device = self.weight.device
        with torch.no_grad():
            
            n = 1000
            for i in range(k):
                v = torch.linspace(*self.r, n)
                X = torch.zeros(n, self.d)
                X[:,i] = v
                Xh = self.forward(X.to(device))

                plot = axes[i]
                plot.set_xlim(self.r)
                plot.plot(v.cpu().numpy(), Xh[:,i].cpu().numpy(), label="Interpolated polynomial activation")

                plot.plot(self.basis.nodes.cpu().numpy(), self.weight[0,i].clone().detach().cpu().numpy(), "x", label="Learned Chebyshev node")

                plot.axvline(x=self.basis.nodes[0].numpy(), linestyle=":", label="Chebyshev x-position")
                for node in self.basis.nodes[1:]:
                    plot.axvline(x=node.numpy(), linestyle=":")
            
                plot.set_xlabel("$x_%d$" % i)

        axes[k//2].legend(bbox_to_anchor=[1.1, -0.1])
        axes[0].set_ylabel("Polynomial output")
        #top[k//2].set_title(title)

        fname = "%s.png" % title
        matplotlib.pyplot.savefig(fname, bbox_inches="tight")
        print("Saved polynomial activations to %s" % fname)
        [plot.cla() for plot in axes]
        [plot.cla() for plot in top]
        return fname
