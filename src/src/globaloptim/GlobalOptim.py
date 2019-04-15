import torch, tqdm, sys, math

class GlobalOptim:

    def __init__(
        self,
        features,
        labels,
        top_n,
        explore,
        exploit,
        table,
        lipo,
        max_retry,
        lr,
        epochs,
        k_folds
    ):
        self.features = features
        self.labels = labels
        self.top_n = top_n
        self.explore = explore
        self.exploit = exploit
        self.table = table
        self.lipo = lipo
        self.max_retry = max_retry
        self.lr = lr
        self.epochs = epochs
        self.k_folds = k_folds

        self.num_evals = 0
        self.init_table()

    def count_evals(self):
        return self.num_evals

    def publish_XY(self):
        X, Y = self._get_XY()
        metascores = self.compute_metascore(Y)
        metascores, I = metascores.sort(descending=True)
        return X[I], Y[I], metascores

    def step(self):
        X, Y = self._get_XY()
        evalnet = self.fit_evalnet(X, Y)

        # Do not use the original Y as it may not be Lipschitz
        Y = self._get_corresponding_outputs(X, evalnet)
        assert len(X) == len(Y)
        
        self._add_to_dataset(self.exploit_Xb(X, Y, evalnet), X, Y, "Exploiting")
        self._add_to_dataset(self._explore_Xb(X, Y), X, Y, " Exploring")

    # === PROTECTED ===

    def exploit_Xb(self, X, Y, evalnet):
        Xb = self._montecarlo_sample_Xb(X)
        self._update_Xb_using_grad_and_lipo(Xb, X, Y, evalnet)
        return Xb

    def init_table(self):
        X = self.lipo.init_sample(max(self.explore, 2))
        X[0] = self.lipo.b
        Y = self._evaluate(X, "Initializing")
        self._insert_XY(X, Y)
        self.table.update_scores(k=0)

    # === ABSTRACT ===

    def create_optimizer(self, parameters, lr):
        '''

        Output:
            optim - torch.optim.Optimizer for the input parameters.

        '''
        raise NotImplementedError

    def evaluate(self, x):
        '''

        Input:
            x - torch Tensor of shape (D), feature selection mask.

        Output:
            y - torch Tensor of shape (1), scores for using x as feature mask.

        '''
        raise NotImplementedError

    def fit_evalnet(self, X, Y):
        '''

        Input:
            X - torch Tensor of shape (N, D), feature selection masks.
            Y - torch Tensor of shape (N), scores for each selection.

        Output:
            evalnet - torch.nn.Module, Lipschitz network that maps
                feature selection to predicted score.

        '''
        raise NotImplementedError

    def neutral_x(self):
        '''

        Output:
            neutral - single sample Tensor of size (1, D) or int, represents the
                center-most sample to start gradient ascent from.

        '''
        raise NotImplementedError

    def compute_metascore(self, Y):
        '''

        Input:
            Y - torch Tensor of size (N, T), output targets.

        Output:
            metascore - torch Tensor of size (N), score for MonteCarlo ranking.
        
        '''
        raise NotImplementedError

    def congregate_scores(self, X):
        '''

        Input:
            X - torch Tensor of size (self.top_n, D), feature selections.

        Output:
            x - torch Tensor of size (self.top_n), congregate selection.

        '''
        raise NotImplementedError

    # === PRIVATE ===

    def _explore_Xb(self, X, Y):
        return self.lipo.sample(self.explore, X, Y)

    def _get_XY(self):
        return self.table.get_XY()

    def _montecarlo_sample_Xb(self, X):
        i = min(X.size(0)-2, self.exploit)
        Xb = X[:i+2].clone()
        Xb[-1] = self.neutral_x()
        Xb[-2] = self.congregate_scores(X[:self.top_n])
        self.table.update_scores(k=i)
        return Xb

    def _update_Xb_using_grad_and_lipo(self, Xb, X, Y, evalnet):
        desc = "Predicting better optima"
        with tqdm.tqdm(range(self.max_retry), ncols=80, desc=desc) as bar:
            for I in self.lipo.check_require_retry(bar, Xb, X, Y):
                # The rows that do not satisfy the LIPO decision rule
                # need to be improved upon by gradient ascent.
                new_Xbs = self._apply_gradient_heuristic(Xb[I], evalnet)
                Xb[I] = self.lipo.clip(new_Xbs)

    def _get_corresponding_outputs(self, X, evalnet):
        batchsize = 32
        its = math.ceil(len(X)/float(batchsize))
        with torch.no_grad():
            # for smaller use of memory, do in batches
            Y = map(evalnet,
                    [X[i*batchsize:(i+1)*batchsize] for i in range(its)]
            )
            return self.compute_metascore(torch.cat(list(Y)))

    def _apply_gradient_heuristic(self, X_exploit, evalnet):
        X_exploit = torch.autograd.Variable(X_exploit, requires_grad=True)
        optim = self.create_optimizer([X_exploit], self.lr)
        optim.zero_grad()
        self._ascend_grad(evalnet, X_exploit)
        optim.step()
        return X_exploit.detach()
        
    def _ascend_grad(self, evalnet, X):
        Y = self.compute_metascore(evalnet(X)).sum()
        X.grad = -torch.autograd.grad(
            [Y], [X],
            create_graph=True,
            only_inputs=True
        )[0]

    def _add_to_dataset(self, Xb, X, Y, taskname):
        I = self.lipo.decision_rule(Xb, X, Y)
        X_targets = Xb[I]
        if len(X_targets) > 0:
            Y_targets = self._evaluate(X_targets, taskname)
            self._insert_XY(X_targets, Y_targets)
        else:
            sys.stderr.write("%s: No samples passed LIPO.\n" % taskname)

    def _evaluate(self, X, taskname):
        N = len(X)
        Y = torch.zeros(N, self.labels)
        bar_count = N*(self.epochs*self.k_folds+1)
        with tqdm.tqdm(range(bar_count), ncols=80) as bar:
            for i in range(N):
                for placeholder in self.evaluate(X[i]):
                    bar.update()
                Y[i] = placeholder # only the last one has value
                top_metascore = self.compute_metascore(Y[:i+1]).max().item()
                bar.set_description("%s (%.5f)" % (taskname, top_metascore))
            self.num_evals += N
            return Y

    def _insert_XY(self, X, Y):
        self.table.insert_XY(X, Y, self.compute_metascore(Y))
