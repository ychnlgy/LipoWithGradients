import torch

class Permute(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, X):
        return X.permute(*self.args)
