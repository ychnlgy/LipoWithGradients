import torch, math

from GlobalOptimizer import *

class NeuralGlobalOptimizer(GlobalOptimizer):

    EPS = 1e-32
    SELECTION = 0.5

    def __init__(self, gradpenalty_weight, *args, prep_visualization=False, **kwargs):
        self.network_retrain_count = 0
        self.gradpenalty_weight = gradpenalty_weight

        self.prep_visualization = prep_visualization
        if prep_visualization:
            self.data_losses = []
            self.test_losses = []
            self.feature_counts = []
            self.store_losses = self._store_losses

        super().__init__(*args, **kwargs)

    def count_network_retrains(self):
        return self.network_retrain_count

    def get_losses(self):
        assert self.prep_visualization
        return (self.data_losses, self.test_losses, self.feature_counts)

    # === PRIVATE ===

    def get_dataset(self):
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
        shortcut = self.lookup_result(x)

        if shortcut is not None:
            return shortcut
        else:
            self.network_retrain_count += 1
            return self.do_expensive_model_eval(
                self.discretize_featuremask(x)
            )

    def discretize_featuremask(self, x):
        return x > NeuralGlobalOptimizer.SELECTION

    def lookup_result(self, x):
        '''

        Output:
            shortcut - float if x has already been evaluated, otherwise None.
            
        '''
        if not len(self.table):
            return None
        
        X, Y = self.get_XY()
        diff = (
            self.discretize_featuremask(x)
        ) != (
            self.discretize_featuremask(X)
        )

        # if there is an entry in which no element is
        # different, then we can use that result.
        mark = diff.long().sum(dim=1) == 0
        if mark.long().sum() > 0:
            shortcut = Y[mark]
            assert len(shortcut) == 1
            return shortcut.item()
        else:
            return None
        
    def do_expensive_model_eval(self, x):
        X_data, Y_data, X_test, Y_test = self.get_dataset()
        X_data *= x.unsqueeze(0).float()
        model = self.make_model(X_data.size(1))
        lossf = self.create_model_lossfunction()
        model.train()
        self.train_model(model, lossf, X_data, Y_data)

        with torch.no_grad():
            model.eval()
            Yh_data = model(X_data).squeeze()
            data_loss = torch.log(
                lossf(Yh_data, Y_data) + NeuralGlobalOptimizer.EPS
            ).item()
            Yh_test = model(X_test).squeeze()
            test_loss = torch.log(
                lossf(Yh_test, Y_test) + NeuralGlobalOptimizer.EPS
            ).item()
            feature_count = x.long().sum().item()
            feature_penalty = self.penalize_featurecount(feature_count)
            self.store_losses(data_loss, test_loss, feature_count)
            return -(data_loss + test_loss + feature_penalty)

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
        X = self.discretize_featuremask(X).float()
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
