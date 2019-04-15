import torch

class LagrangeBasis(torch.nn.Module):

    @staticmethod
    def create(nodes, eps=1e-8):
        '''

        Description:
            Entry point for instantiating the LagrangeBasis instance.
            We wish to create the polynomial function based on the
            coordinates of the input nodes.

        Input:
            nodes - torch Tensor of shape (n), the x-coordinates of
                the positions we wish to fit an polynomial.
            eps - float, for numerical stability of the denominator.

        Output:
            p - LagrangeBasis, the n-basis functions that will create
                a polynomial that intersects each x-coordinate exactly.

        '''
        n = len(nodes)

        m_i = (1-torch.eye(n)).byte()
        m_v = torch.arange(n).view(1, n).repeat(n, 1)
        m_s = m_v[m_i].view(n, n-1)

        # xm is shape (1, 1, n, n-1, 1)
        #   - index[2]: number of basis functions.
        #   - index[3]: points involved per basis function.
        xm = nodes[m_s].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        xj = nodes.view(1, n, 1, 1)

        # dn is shape (1, 1, n, 1)
        denominator = (xj-xm).prod(dim=3) + eps
        return LagrangeBasis(xm, denominator)

    def forward(self, X):

        '''

        Input:
            X - torch Tensor of size (N, D, *)

        Output:
            L - torch Tensor of size (N, D, n, *), where n is the
                number of nodes for the Lagrange basis.

        '''
        N = X.size(0)
        D = X.size(1)
        shape = X.shape[2:] # for restoring size at the end
        
        X = X.view(N, D, 1, 1, -1)
        out = (X - self.xm).prod(dim=3)/self.dn
        return out.view(N, D, self.n, *shape)

    # === PROTECTED ===

    def __init__(self, xm, denominator):
        super().__init__()
        self.register_buffer("xm", xm)
        self.xm.requires_grad = False
        self.register_buffer("dn", denominator)
        self.dn.requires_grad = False
        self.n = self.dn.size(2)

if __name__ == "__main__":

    def equals_float(v, w):
        eps = 1e-6
        return (v-w).norm() < eps

    # Test: {x1 = -1, x2 = 0, x3 = 1}
    nodes = torch.Tensor([-1, 0, 1])
    basis = LagrangeBasis.create(nodes)

    # We expect:
    #   l1(x) = x(x-1)/2
    #   l2(x) = (x+1)(x-1)/-1
    #   l3(x) = (x+1)x/2

    assert equals_float(basis.xm, torch.FloatTensor([
        [ 0, 1],
        [-1, 1],
        [-1, 0]
    ]).view(1, 1, 3, 2, 1)) # xm is what we subtract

    assert equals_float(basis.dn, torch.FloatTensor([
        [2, -1, 2]
    ]).view(1, 1, 3, 1)) # dn is what we divide
