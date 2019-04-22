import torch

class LagrangeBasis(torch.nn.Module):

    @staticmethod
    def create(nodes):
        n = len(nodes)

        m_i = (1-torch.eye(n)).byte()
        m_v = torch.arange(n).view(1, n).repeat(n, 1).long()
        m_s = m_v[m_i].view(n, n-1)

        xm = nodes[m_s].unsqueeze(0)
        xj = nodes.view(1, n, 1)

        denominator = (xj-xm).prod(dim=2)
        return LagrangeBasis(xm, denominator, nodes)

    def __init__(self, xm, denominator, nodes):
        super().__init__()
        self.nodes = nodes
        self.register_buffer("xm", xm)
        self.xm.requires_grad = False
        self.register_buffer("dn", denominator)
        self.dn.requires_grad = False

    def forward(self, X):

        '''

        Input:
            X - torch Tensor of size (N, *, D)

        Output:
            L - torch Tensor of size (N, *, D, n), where n is the
                number of nodes for the Lagrange basis.

        '''

        device = X.device
        shape = X.size()
        X = X.view(-1, 1, 1)
        out = (X-self.xm).prod(dim=-1)/self.dn
        return out.view(*shape, out.size(-1))

    def visualize(self, plot):
        pass
        
