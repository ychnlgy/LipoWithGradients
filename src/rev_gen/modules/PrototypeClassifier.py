import torch, math

DEFAULT = torch.nn.CosineSimilarity(dim=1)

class PrototypeClassifier(torch.nn.Module):

    def __init__(self, features, classes, similarity=DEFAULT):
        super().__init__()
        self.C = classes
        self.D = features
        self.weight = torch.nn.Parameter(torch.zeros(1, classes, features))
        self.similarity = similarity

        self.reset_parameters()

    def forward(self, X):
        D = X.size(-1)
        T = X.contiguous().view(-1, D)
        N = T.size(0)
        assert D == self.D
        T = T.view(N, 1, D).repeat(1, self.C, 1).view(N*self.C, D)
        P = self.weight.repeat(N, 1, 1).view(N*self.C, D)
        S = self.similarity(T, P)
        return S.view(*X.shape[:-1], self.C)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
