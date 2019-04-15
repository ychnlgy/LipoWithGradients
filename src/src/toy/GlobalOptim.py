import torch, tqdm, numpy, os

import src

from . import datasets, get_type

class Dataset:

    def __init__(self, X, Y, folds):
        self.X = X
        self.Y = Y
        self.k = folds

    def create_train_valid_iterator(self):
        return self.split_train_valid(self.X, self.Y, self.k)

    def split_train_valid(self, X, Y, k):
        raise NotImplementedError

class RandomDataset(Dataset):

    def split_train_valid(self, X, Y, k):
        N = X.size(0)
        VALIDATION_SIZE = N//k      
        I = src.tensortools.rand_indices(X.size(0))
        for p in range(k):
            i = VALIDATION_SIZE * p
            j = VALIDATION_SIZE *(p+1)
            I_valid = I[i:j]
            I_train = torch.cat([I[:i], I[j:]], dim=0)
            yield X[I_train], Y[I_train], X[I_valid], Y[I_valid]

class SubjectDataset(Dataset):

    def __init__(self, train_X, train_Y, U, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_X = train_X
        self.train_Y = train_Y
        self.U = U

    def split_train_valid(self, X, Y, k):
        for (X_train, Y_train), (X_valid, Y_valid) in src.cross_validation.k_fold(self.U, X, Y, k):
            X_train = numpy.concatenate([X_train, self.train_X])
            Y_train = numpy.concatenate([Y_train, self.train_Y])
            X_data, Y_data, X_test, Y_test = tuple(map(torch.from_numpy, [
                X_train, Y_train, X_valid, Y_valid
            ]))
            yield X_data.float(), Y_data.long(), X_test.float(), Y_test.long()

class MseScorer(src.tensortools.Scorer):

    def reset(self):
        self.s = 0.0
        self.n = 0.0

    def update(self, yh, y):
        self.s += torch.nn.functional.mse_loss(yh.mean(dim=-1), y, reduction="sum").item()
        self.n += len(yh)

    def peek(self):
        return 1-self.s/self.n

    def calc_sens(self):
        raise AssertionError

    def calc_spec(self):
        raise AssertionError

class GlobalOptim(src.globaloptim.NeuralGlobalOptim):

    def __init__(self, features, dataname, modeltype, modeltier, gradpenalty):
        self._dataset = None
        self.truemask = None
        self.model_lossf = None
        self.dataname = dataname
        self.modeltype = modeltype
        self.modeltier = modeltier
        self.gradpenalty = gradpenalty
        
        super().__init__(
            feature_penalty = 1e-2,
            mutation_rate = 0.5,
            features = features,
            labels = 2,
            top_n = 10,
            explore = 4,
            exploit = 11,
            table = src.globaloptim.AdaptiveDataTable(
                capacity = 1200,
                reduced_size = 1000,
                features = features,
                labels = 2,
                montecarlo_c = 1e-3
            ),
            lipo = src.globaloptim.Lipo(
                k = 1.1, # leave some error room for 1-Lipschitz network
                d = features,
                a = -1,
                b = +1
            ),
            max_retry = 1000,
            lr = 0.1, # should experiment with smaller learning rates
            epochs = 200,
            k_folds = 5
        )

    # === PROTECTED ===

    def init_table(self):
        self._init_dataset()
        super().init_table()
        
    def get_dataset(self):
        '''

        Output:
            X_data - torch Tensor of size (N, D), input features.
            Y_data - torch Tensor of size (N, T), desired output targets.
            X_test - torch Tensor of size (M, D), input features that may or
                may not exist in X_data.
            Y_test - torch Tensor of size (M, T), expected output for X_test.

        '''
        return self._dataset.create_train_valid_iterator()

    def make_model(self, D):
        '''

        Output:
            model - torch.nn.Module, untrained.

        '''
        return get_type(self.modeltype, self.modeltier)(D)
    
    def train_model(self, model, X, Y):
        '''

        Description:
            Trains the model in place.

        Input:
            model - see self.make_model (note: training mode is activated).
            X - see self.get_dataset: X_data.
            Y - see self.get_dataset: Y_data.

        Output:
            iterator of None - for updating the progress bar.

        '''
        dataloader = src.tensortools.dataset.create_loader([X, Y], batch_size=8, shuffle=True)
        
        optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100])
        for e in range(self.epochs):
            for x, y in dataloader:
                yh = model(x)
                loss = self.model_lossf(yh, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
            sched.step()
            yield
            
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
        dataloader = src.tensortools.dataset.create_loader([X, Y], batch_size=8)
        self.scorer.reset()
        for x, y in dataloader:
            yh = model(x)
            self.scorer.update(yh, y)
        return 1-self.scorer.peek() # the algorithm finds the minimum.

    def create_evalnet(self, D):
        '''

        Output:
            evalnet - torch.nn.Module, network for mapping feature selections
                to validation score and penalty on number of features.

        '''
        
        return torch.nn.Sequential(
            torch.nn.Linear(D, 32),

            src.modules.ResNet(
                src.modules.ResBlock(
                    block = torch.nn.Sequential(
                        src.modules.PrototypeClassifier(32, 32),
                        src.modules.polynomial.Activation(32, n_degree=4),
                        torch.nn.Linear(32, 32)
##                        torch.nn.ReLU(),
##                        torch.nn.Linear(128, 128),
                    ),
##                    activation = torch.nn.ReLU(),
                ),
                src.modules.ResBlock(
                    block = torch.nn.Sequential(
                        src.modules.PrototypeClassifier(32, 32),
                        src.modules.polynomial.Activation(32, n_degree=4),
                        torch.nn.Linear(32, 32)
##                        torch.nn.ReLU(),
##                        torch.nn.Linear(128, 128),
                    ),
##                    activation = torch.nn.ReLU(),
                ),
##                src.modules.ResBlock(
##                    block = torch.nn.Sequential(
##                        #src.modules.PrototypeClassifier(32, 32),
##                        #src.modules.polynomial.Activation(32, n_degree=6),
##                        #torch.nn.Linear(32, 32)
##                        torch.nn.ReLU(),
##                        torch.nn.Linear(128, 128),
##                    ),
##                   activation = torch.nn.ReLU(),
##                ),
            ),

            torch.nn.Linear(32, 2)
        )

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
        dataloader = src.tensortools.dataset.create_loader([X, Y], batch_size=8, shuffle=True)
        epoch = 200
        lossf = torch.nn.MSELoss()
        optim = torch.optim.SGD(evalnet.parameters(), lr=0.01, momentum=0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100])

        avg = src.util.MovingAverage(momentum=0.99)
        bar = tqdm.tqdm(range(epoch), ncols=80)
        for e in bar:
            for x, y in dataloader:
                yh = evalnet(x)
                loss = lossf(yh, y)
                full_loss = loss + src.algorithm.grad_penalty.lipschitz_max_grad(evalnet, x, X, Y)
                optim.zero_grad()
                full_loss.backward()
                optim.step()

                avg.update(loss.item())
                bar.set_description("Evalnet loss %.4f" % avg.peek())
            sched.step()

    # === PRIVATE ===

    def _create_dataset(self):
        
        if os.path.isdir(self.dataname):
            model_lossf = torch.nn.CrossEntropyLoss()
            scorer = src.tensortools.Scorer()
            out = self._create_dataset_winterlight()
        
        else:

            model_lossf = lambda yh, y: torch.nn.functional.mse_loss(yh.mean(dim=-1), y)
            scorer = MseScorer()
            
            (X, Y), _, _ = datasets.create_from_str(
                self.dataname,
                N = 200,
                D = self.features,
                noise = 0
            )
            self._set_true_mask()
            self.k_folds = 2
            out = RandomDataset(X, Y, folds=self.k_folds)
        
        return model_lossf, scorer, out

    def _create_dataset_winterlight(self):
        (db_X, db_Y, db_U, _), _, (ha_X, ha_Y, _, _) = src.winterlightlabs.load(self.dataname)
        return SubjectDataset(ha_X, ha_Y, db_U, db_X, db_Y, self.k_folds)

    def _set_true_mask(self):
        self.truemask = numpy.zeros(self.features, dtype=numpy.uint8)
        for i in datasets.TRUE_MASK:
            self.truemask[i] = 1

    def _init_dataset(self):
        self.model_lossf, self.scorer, self._dataset = self._create_dataset()
