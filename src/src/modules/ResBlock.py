import torch

IDENTITY = torch.nn.Sequential()

class ResBlock(torch.nn.Module):

    def __init__(self, block, shortcut=IDENTITY, activation=IDENTITY, allow_reverse=False):
        super(ResBlock, self).__init__()
        self.bk = block
        self.sc = shortcut
        self.ac = activation
        self.allow_reverse = allow_reverse
        
    def forward(self, X):
        return self.ac(self.bk(X) + self.sc(X))
