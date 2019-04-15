import torch

from .GlobalOptim import GlobalOptim
from .Lipo import Lipo

SELECTION_THRESHOLD = 0

class NeuralGlobalOptim(GlobalOptim):

    def __init__(
        self,
        feature_penalty,
        mutation_rate,
        selection_threshold = SELECTION_THRESHOLD,
        **kwargs
    ):
        self.feature_penalty = feature_penalty
        self.mutation_rate = mutation_rate
        self.selection_threshold = selection_threshold

        super().__init__(**kwargs)

        self.neg_sampler = Lipo(
            k = self.lipo.k,
            d = self.lipo.d,
            a = self.lipo.a,
            b = selection_threshold
        )

    def publish_XY(self):
        X, Y, metascores = super().publish_XY()
        X = self._discretize(X)
        return X, Y, metascores

    # === PROTECTED ===

    def exploit_Xb(self, X, Y, evalnet):
        Xb = super().exploit_Xb(X, Y, evalnet)
        M = self._discretize(Xb)
        mutation_threshold = self.mutation_rate/(1+M.float().sum(dim=1).unsqueeze(1))
        # Only selected features have a chance of mutation.
        I = M & (torch.rand_like(Xb) <= mutation_threshold)
        Xb[I] = self.neg_sampler.sample(Xb.size(0), X, Y)[I]
        return Xb

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
            y - torch Tensor of shape (T), scores for using x as feature mask.

        '''
        mask = self._discretize(x)
        penalty = self._penalize_features(mask)
        score = n = 0.0
        for X_data, Y_data, X_test, Y_test in self.get_dataset():
            X_data, X_test = self._apply_mask(mask, [X_data, X_test])
            model = self.make_model(X_data.size(1))

            model.train()
            for placeholder in self.train_model(model, X_data, Y_data):
                yield

            model.eval()
            with torch.no_grad():
                score += self.eval_model(model, X_test, Y_test)
                n += 1.0

        # negative since LIPO finds global max.
        yield -torch.FloatTensor([score/n, penalty])

    def fit_evalnet(self, X, Y):
        '''

        Input:
            X - torch Tensor of shape (N, D), feature selection masks.
            Y - torch Tensor of shape (N, T), scores for each selection.

        Output:
            evalnet - torch.nn.Module, Lipschitz network that maps
                feature selection to predicted score.

        '''
        evalnet = self.create_evalnet(X.size(1))
        
        evalnet.train()
        self.train_evalnet(evalnet, X, Y)

        evalnet.eval()
        return evalnet

    def neutral_x(self):
        '''

        Output:
            neutral - single sample Tensor of size (1, D) or int, represents the
                center-most sample to start gradient ascent from.

        '''
        return self.selection_threshold

    def compute_metascore(self, Y):
        '''

        Input:
            Y - torch Tensor of size (N, T), output targets.

        Output:
            metascore - torch Tensor of size (N), score for MonteCarlo ranking.
        
        '''
        return Y.sum(dim=1)

    def congregate_scores(self, X):
        '''

        Input:
            X - torch Tensor of size (self.top_n, D), feature selections.

        Output:
            x - torch Tensor of size (self.top_n), congregate selection.

        '''
        return (self._discretize(X).long().sum(dim=0) == len(X)).float()

    # === ABSTRACT ===

    def get_dataset(self):
        '''

        Output:
            iterator of:
                X_data - torch Tensor of size (N, D), input features.
                Y_data - torch Tensor of size (N, T), desired output targets.
                X_test - torch Tensor of size (M, D), input features that may or
                    may not exist in X_data.
                Y_test - torch Tensor of size (M, T), expected output for X_test.

        '''
        raise NotImplementedError

    def make_model(self, D):
        '''

        Output:
            model - torch.nn.Module, untrained.

        '''
        raise NotImplementedError

    def train_model(self, model, X, Y):
        '''

        Description:
            Trains the model in place.

        Input:
            model - see self.make_model (note: training mode is activated).
            X - see self.get_dataset: X_data.
            Y - see self.get_dataset: Y_data.

        Output:
            None

        '''
        raise NotImplementedError

    def eval_model(self, model, X, Y):
        '''

        Description:
            Calculates a single floating number that represents the performance
            of the model on a validation or test set.

        Input:
            model - see self.make_model (note: evaluation mode is activated).
            X - see self.get_dataset: X_test.
            Y - see self.get_dataset: Y_test.

        Output:
            score - float, score of the model on the validation or test set.

        '''
        raise NotImplementedError

    def create_evalnet(self, D):
        '''

        Output:
            evalnet - torch.nn.Module, network for mapping feature selections
                to validation score and penalty on number of features.

        '''
        raise NotImplementedError

    def train_evalnet(self, evalnet, X, Y):
        '''

        Description:
            Trains the evaluation network in place.

        Input:
            evalnet - see self.create_evalnet (note: training mode is activated).
            X - torch Tensor of size (N, D), feature masks.
            Y - torch Tensor of size (N, T), scores for using each feature mask.

        Output:
            None

        '''
        raise NotImplementedError
            
    # === PRIVATE ===

    def _penalize_features(self, mask):
        raw_score = mask.float().sum().item()/torch.numel(mask)
        return self.feature_penalty * raw_score

    def _discretize(self, X):
        return X > self.selection_threshold

    def _apply_mask(self, mask, targets):
        mask = mask.float().unsqueeze(0)
        return [mask * target for target in targets]
    

    
    
