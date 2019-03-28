import torch, math, statistics, os

from GlobalOptimizer import *

class NeuralGlobalOptimizer(GlobalOptimizer):

    SELECTION = 0.5

    def __init__(
        self,
        gradpenalty_weight,
        mutation_rate,
        expected_train_loss,
        featurepenalty_frac,
        *args,
        prep_visualization=False,
        **kwargs
    ):
        self.network_retrain_count = 0
        self.gradpenalty_weight = gradpenalty_weight
        self.mutation_rate = mutation_rate
        self.expected_train_loss = expected_train_loss
        self.featurepenalty_frac = featurepenalty_frac

        self.prep_visualization = prep_visualization
        if prep_visualization:
            self.data_losses = []
            self.test_losses = []
            self.feature_counts = []
            self.store_losses = self._store_losses

        self._dataset = None

        super().__init__(*args, **kwargs)

    def count_network_retrains(self):
        return self.network_retrain_count

    def get_losses(self):
        assert self.prep_visualization
        out = tuple(
            map(
                statistics.mean,
                (self.data_losses, self.test_losses, self.feature_counts)
            )
        )
        self.data_losses = []
        self.test_losses = []
        self.feature_counts = []
        return out

    # === PRIVATE ===

    def exploit_Xb(self, X, Y, evalnet):
        X = super().exploit_Xb(X, Y, evalnet)

        # NOTE: Try to stay away from evolutionary methods
        #R1 = torch.rand_like(X)
        #R2 = torch.rand_like(X)
        #I = R1 <= self.mutation_rate
        #N = X.size(0)
        #X[I] = self.lipo.sample(N)[I]
        return X

    def get_dataset(self):
        if self._dataset is None:
            dpath = self.get_dataset_path()
            if not os.path.isfile(dpath):
                self._dataset = self.create_dataset()
                torch.save(self._dataset, dpath)
            else:
                self._dataset = torch.load(dpath)
        return [d.clone() for d in self._dataset]

    def get_dataset_path(self):
        raise NotImplementedError

    def create_dataset(self):
        raise NotImplementedError

    def make_model(self, D):
        raise NotImplementedError

    def train_model(self, model, lossf, X, Y):
        raise NotImplementedError

    def create_model_lossfunction(self):
        raise NotImplementedError

    def penalize_featurecount(self, count):
        raise NotImplementedError

    def create_evalnet(self, D):
        raise NotImplementedError

    def train_evalnet(self, evalnet, X, Y):
        raise NotImplementedError

    def store_losses(self, *args):
        return

    def _store_losses(self, data_loss, test_loss, feature_count):
        self.data_losses.append(data_loss)
        self.test_losses.append(test_loss)
        self.feature_counts.append(feature_count)

    def create_optimizer(self, parameters, lr):
        '''

        Output:
            optim - torch.optim.Optimizer for the input parameters.

        '''
        return torch.optim.SGD(parameters, lr=lr)

    def evaluate(self, x):
        '''

        Input:
            x - torch Tensor of shape (D), feature selection mask.

        Output:
            y - torch Tensor of shape (1), scores for using x as feature mask.

        '''
        self.network_retrain_count += 1
        x = NeuralGlobalOptimizer.discretize_featuremask(x)
        X_data, Y_data, X_test, Y_test = self.get_dataset()
        mask = x.unsqueeze(0).float()
        X_data *=mask
        X_test *= mask
        D = X_data.size(1)
        model = self.make_model(D)
        lossf = self.create_model_lossfunction()
        model.train()
        self.train_model(model, lossf, X_data, Y_data)

        with torch.no_grad():
            model.eval()
            Yh_data = model(X_data).squeeze()
            data_loss = lossf(Yh_data, Y_data).item()
            Yh_test = model(X_test).squeeze()
            test_loss = lossf(Yh_test, Y_test).item()
            feature_count = mask.sum().item()
            feature_penalty = self.penalize_featurecount(feature_count, D)
            self.store_losses(data_loss, test_loss, feature_count)
            return -(data_loss + test_loss + feature_penalty)

    @staticmethod
    def discretize_featuremask(x):
        return x > NeuralGlobalOptimizer.SELECTION

    def fit_evalnet(self, X, Y):
        '''

        Input:
            X - torch Tensor of shape (N, D), feature selection masks.
            Y - torch Tensor of shape (N), scores for each selection.

        Output:
            evalnet - torch.nn.Module, Lipschitz network that maps
                feature selection to predicted score.

        '''
        self._used_gradpenalty = False
        evalnet = self.create_evalnet(X.size(1))
        evalnet.train()
        self.train_evalnet(evalnet, X, Y)
        assert self._used_gradpenalty
        evalnet.eval()
        return evalnet

    def grad_penalty(self, evalnet, X):
        self._used_gradpenalty = True
        X = torch.autograd.Variable(X, requires_grad=True)
        Y = evalnet(X).sum()
        T = list(evalnet.parameters()) + [X]
        grads = torch.autograd.grad([Y], T, create_graph=True)
        gp = sum(map(NeuralGlobalOptimizer.lipschitz1_loss, grads))
        return gp * self.gradpenalty_weight

    @staticmethod
    def lipschitz1_loss(g):
        return torch.nn.functional.relu(g.abs()-1).sum()
