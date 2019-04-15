import torch

MAX_RESAMPLE = 10000

class Lipo:

    def __init__(self, k, d, a, b):
        assert b > a
        self.k = k
        self.d = d
        self.a = a
        self.b = b

    def scale(self, X):
        "Assumes X is a matrix of values in [0, 1]."
        return X * (self.b-self.a) +self.a

    def sample(self, n, X, Y):
        Xb = self.init_sample(n)
        for I in self.check_require_retry(range(MAX_RESAMPLE), Xb, X, Y):
            Xb[I] = self.init_sample(n)[I]
        return Xb

    def check_require_retry(self, iterator, Xb, X, Y):
        for _ in iterator:
            I = ~self.decision_rule(Xb, X, Y)
            if I.long().sum() == 0:
                break
            else:
                yield I

    def decision_rule(self, Xb, X, Y):
        Xb = Xb.unsqueeze(1)
        X  = X.unsqueeze(0)
        Y  = Y.view(1, -1)
        scores = Y + self.k * (X - Xb).norm(dim=2, p=2)
        V, _ = scores.min(dim=1)
        return V > Y.max()

    def clip(self, X):
        X = X.clone()
        X[X<self.a] = self.a
        X[X>self.b] = self.b
        return X

    def init_sample(self, n):
        return self.scale(torch.rand(n, self.d))
