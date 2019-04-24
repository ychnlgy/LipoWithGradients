import torch

class Contiguous(torch.nn.Module):

    def forward(self, X):
        return X.contiguous()
