import torch, math

DEFAULT_C = math.sqrt(2)

class MonteCarloTree:

    def __init__(self, score_i, n_i, output_i, c=DEFAULT_C):
        self.s_i = score_i
        self.n_i = n_i
        self.o_i = output_i
        self.c = c
        self.N = 0

    def simulate(self, X):
        self.N += 1
        X[:,self.o_i] = self._score(X)

    # === PRIVATE ===

    def _score(self, X):
        s = X[:,self.s_i]
        n = X[:,self.n_i]
        return s/n + self.c*torch.sqrt(math.log(self.N)/n)
