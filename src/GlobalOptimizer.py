import math, torch, numpy, tqdm, sys

from MovingAverage import MovingAverage

class EvalNet(torch.nn.Module):

    def __init__(self, net, rev):
        super().__init__()
        self.net = net
        self.rev = rev

    def forward(self, X):
        return self.net(X)

    def reverse(self, Y):
        return self.rev(Y)

class GlobalOptimizer:

    '''

    Notes:
        This algorithm depends on LIPO for selecting points to evaluate during
        the search of the global maximum. I do not understand the theory of LIPO
        enough to tamper with it, therefore we need to change the goals of
        optimization from standard gradient descent and minimization into
        gradient ascent and maximization.

    '''

    def __init__(self, features, explore, exploit, table, lipo, max_retry, lr, savepath):
        self.table = table
        self.lipo = lipo
        self.features = features
        self.explore = explore
        self.exploit = exploit
        self.max_retry = max_retry
        self.lr = lr
        self.savepath = savepath
        self.num_evals = 0

        self.initialize_table()

    def count_evals(self):
        return self.num_evals

    def step(self):
        X, Y = self.get_XY()
        evalnet = self.fit_evalnet(X, Y)
        args = (X, Y, evalnet)
        self.add_to_dataset(self.exploit_Xb(*args), *args, "Exploiting")
        self.add_to_dataset(self.explore_Xb(*args), *args, " Exploring")
        self.table.update_scores(k=self.exploit)

    def save(self):
        torch.save(self.publish_XY(), self.savepath)

    def publish_XY(self):
        X, Y = self.get_XY()
        Y, I = Y.sort(descending=True)
        return X[I], Y

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

    # === PRIVATE ===

    def get_XY(self):
        '''

        Output:
            X - torch Tensor of size (N, D), feature masks sorted by score.
            Y - torch Tensor of size (N), actual scores for each feature mask.

        '''
        X = self.table.get_X()
        Y = self.table.get_Y()
        return (X, Y)

    def initialize_table(self):
        X = torch.eye(self.features+2, self.features)
        X[-1] = 1
        Y = self._evaluate(X, "Random initialization")
        self.insert_xy(X, Y)
        self.table.update_scores(k=0)

    def _evaluate(self, X, taskname):
        N = len(X)
        Y = torch.zeros(N)
        bar = tqdm.tqdm(range(N), ncols=80)
        for i in bar:
            Y[i] = self.evaluate(X[i])
            bar.set_description("%s (%.3f)" % (taskname, Y[:i+1].max().item()))
        self.num_evals += N
        return Y

    def explore_Xb(self, X, Y, evalnet):
        return self.lipo.sample(self.explore)

    def neutral_x(self):
        raise NotImplementedError

    def exploit_Xb(self, X, Y, evalnet):
        Xb = X[:self.exploit].clone()
        Xb[-1] = self.neutral_x()
        with torch.no_grad():
            Xb[-2] = evalnet.reverse(Y.max().unsqueeze(0))

        print((Xb[-2]>0.5).numpy(), Y.max().item())

        for i in range(self.max_retry):

            # The rows that do not satisfy the LIPO decision rule
            # need to be improved upon by gradient ascent.
            I = ~self.lipo.decision_rule(Xb, X, Y)
            
            if I.long().sum() == 0:
                break

            X_exploit = torch.autograd.Variable(Xb[I], requires_grad=True)
            optim = self.create_optimizer([X_exploit], self.lr)

            optim.zero_grad()
            self.exploit_grads(evalnet, X_exploit)
            optim.step()
            Xb[I] = X_exploit.detach()
        return Xb

    def add_to_dataset(self, Xb, X, Y, evalnet, taskname):
        # The rows that now satisfy the LIPO decision rule
        # get to be evaluated.
        X_targets = Xb[self.lipo.decision_rule(Xb, X, Y)].detach()
        
        if len(X_targets) > 0:
            Y_targets = self._evaluate(X_targets, taskname)
            self.insert_xy(X_targets, Y_targets)
        else:
            print("Nothing to optimize!")

    def exploit_grads(self, evalnet, X):
        Y = evalnet(X).sum()
        X.grad = -torch.autograd.grad([Y], [X], create_graph=True, only_inputs=True)[0]

    def insert_xy(self, X, Y):
        self.table.insert_xy(X, Y)
        
class GlobalOptimizationTable:

    METAINDEX_SCORE = 0
    METAINDEX_N = 1
    METAINDEX_I = 2
    METAINDEX_Y = 3

    def __init__(self, capacity, features, reduced_size, montecarlo_c):
        self.capacity = capacity
        self.X = torch.zeros(capacity, features)
        self.metadata = torch.zeros(capacity, 4)
        self.reduced_size = reduced_size
        self.montecarlo = MonteCarloTree(
            score_i = GlobalOptimizationTable.METAINDEX_Y,
            n_i = GlobalOptimizationTable.METAINDEX_N,
            output_i = GlobalOptimizationTable.METAINDEX_SCORE,
            c = montecarlo_c
        )
        self.i = 0

    def __len__(self):
        return self.i

    def insert_xy(self, X, Y):
        addition = X.size(0)
        self.check_capacity(addition)
        j = self.i + addition
        self.X[self.i:j] = X
        self.metadata[self.i:j] = self.new_meta_entry(Y)
        self.i = j

    def update_scores(self, k):
        self.montecarlo.simulate(self.metadata[:self.i])
        self.sort_metadata()
        if k > 0:
            self.metadata[:k,GlobalOptimizationTable.METAINDEX_N] += 1

    def get_Y(self):
        return self.get_metadata(GlobalOptimizationTable.METAINDEX_Y)

    def get_X(self):
        I = self.get_metadata(GlobalOptimizationTable.METAINDEX_I)
        return self.X[I.long()]

    # === PRIVATE ===

    def new_meta_entry(self, Y):
        N = len(Y)
        entry = torch.zeros(N, 4)
        entry[:,GlobalOptimizationTable.METAINDEX_N] = 1
        new_I = torch.arange(self.i, self.i+N).float()
        entry[:,GlobalOptimizationTable.METAINDEX_I] = new_I
        entry[:,GlobalOptimizationTable.METAINDEX_Y] = Y
        return entry
    
    def check_capacity(self, addition):
        if self.i + addition > self.capacity:
            self.reduce_size()

    def reduce_size(self):
        sys.stderr.write(
            "[Global optimization table]: reduced size from %d to %d.\n" % (
                self.i, self.reduced_size
            )
        )
        self.swap_and_sort_metadata(GlobalOptimizationTable.METAINDEX_Y)
        self.i = self.reduced_size
        self.X[:self.i] = self.get_X()
        self.metadata[:self.i,GlobalOptimizationTable.METAINDEX_I] = torch.arange(self.i).float()

        # for sanity's sake:
        self.metadata[self.i:] = 0
        self.X[self.i:] = 0

    def swap_and_sort_metadata(self, i):
        self.swap(i)
        self.sort_metadata()
        self.swap(i)

    def swap(self, i):
        j = GlobalOptimizationTable.METAINDEX_SCORE
        self.metadata[:,j], self.metadata[:,i] = self.metadata[:,i], self.metadata[:,j]

    def get_metadata(self, i):
        return self.metadata[:self.i,i]

    def sort_metadata(self):
        _, i = self.get_metadata(GlobalOptimizationTable.METAINDEX_SCORE).sort(descending=True)
        self.metadata[:self.i] = self.metadata[i]

class MonteCarloTree:

    def __init__(self, score_i, n_i, output_i, c):
        self.s_i = score_i
        self.n_i = n_i
        self.o_i = output_i
        self.c = c
        self.N = 0

    def simulate(self, X):
        self.N += 1
        X[:,self.o_i] = self.score(X)

    # === PRIVATE ===

    def score(self, X):
        s = X[:,self.s_i]
        n = X[:,self.n_i]
        return s/n + self.c*torch.sqrt(math.log(self.N)/n)

class Lipo:

    def __init__(self, k, d, a, b):
        self.k = k
        self.d = d
        self.a = a
        self.b = b

    def sample(self, n):
        return torch.rand(n, self.d)*(self.b - self.a)+self.a

    def decision_rule(self, Xb, X, Y):
        Xb = Xb.unsqueeze(1)
        X  = X.unsqueeze(0)
        Y  = Y.view(1, -1)
        scores = Y + self.k * (X - Xb).norm(dim=2, p=2)
        V, _ = scores.min(dim=1)
        return V > Y.max()
