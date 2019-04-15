import torch, sys

import src

METASIZE = 4
META_S = 0
META_N = 1
META_I = 2
META_Y = 3

class AdaptiveDataTable:

    def __init__(
        self,
        capacity,
        reduced_size,
        features,
        labels,
        montecarlo_c,
        verbose = True
    ):
        self.capacity = capacity
        self.reduced_size = reduced_size
        self.features = features
        self.labels = labels
        self.verbose = verbose

        self.X = torch.zeros(capacity, features)
        self.Y = torch.zeros(capacity, labels)
        self.metatable = torch.zeros(capacity, METASIZE)

        self.i = 0

        self.montecarlo = src.algorithm.MonteCarloTree(
            score_i = META_Y,
            n_i = META_N,
            output_i = META_S,
            c = montecarlo_c
        )

    def update_scores(self, k):
        if k > 0:
            self.metatable[:min(k, self.i), META_N] += 1
        self.montecarlo.simulate(self.metatable[:self.i])
        self._sort_metatable()

    def __len__(self):
        return self.i

    def insert_XY(self, X, Y, metascores):
        addition = X.size(0)
        self._check_capacity(addition)
        j = self.i + addition
        slc = slice(self.i, j)
        self.X[slc] = X
        self.Y[slc] = Y
        self._update_metatable(slc, metascores)
        self.i = j

    def get_XY(self):
        I = self._get_metadata(META_I).long()
        return self.X[I], self.Y[I]

    # === PRIVATE ===

    def _get_metadata(self, i):
        return self.metatable[:self.i,i]

    def _update_metatable(self, slc, metascores):
        self.metatable[slc, META_N] = 1
        self.metatable[slc, META_I] = torch.arange(slc.start, slc.stop).float()
        self.metatable[slc, META_Y] = metascores

    def _check_capacity(self, addition):
        if self.i + addition > self.capacity:
            self._reduce_size()

    def _reduce_size(self):
        self._report_reduce_size()
        self._sort_by_metascores()
        self.i = self.reduced_size
        self.X[:self.i], self.Y[:self.i] = self.get_XY()
        self._update_metatable_with_reduced_size()

    def _sort_by_metascores(self):
        self._swap_metascore()
        self._sort_metatable()
        self._swap_metascore()

    def _swap_metascore(self):
        (
            self.metatable[:,META_Y],
            self.metatable[:,META_S]
        ) = (
            self.metatable[:,META_S],
            self.metatable[:,META_Y]
        )

    def _sort_metatable(self):
        _, i = self._get_metadata(META_S).sort(descending=True)
        self.metatable[:self.i] = self.metatable[i]

    def _update_metatable_with_reduced_size(self):
        self.metatable[:self.i, META_I] = torch.arange(self.i).float()
        # for sanity's sake:
        self.X[self.i:] = 0
        self.Y[self.i:] = 0
        self.metatable[self.i:] = 0

    def _report_reduce_size(self):
        if self.verbose:
            sys.stderr.write(
                "[Global optimization table]: reduced size from %d to %d.\n" % (
                    self.i, self.reduced_size
                )
            )
