import torch, tqdm, math, numpy
import torch.utils.data

from NeuralGlobalOptimizer import *
from MovingAverage import MovingAverage
import modules

from main import main

class Equal4(NeuralGlobalOptimizer):

    D = 20
    TRUE_D = 4

    def get_dataset(self):
        N = 500
        X = torch.rand(N*2, Equal4.D)
        Y = X[:,:Equal4.TRUE_D].sum(dim=1)
        return X[:N], Y[:N], X[N:], Y[N:]

    def make_model(self, D):
        return torch.nn.Linear(D, 1)

    def train_model(self, model, lossf, X, Y):
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        epochs = 200

        optim = torch.optim.Adam(model.parameters())
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100])

        for epoch in range(epochs):

            for x, y in dataloader:
                yh = model(x).squeeze()
                loss = lossf(yh, y)
                optim.zero_grad()
                loss.backward()
                optim.step()

            sched.step()

    def create_model_lossfunction(self):
        return torch.nn.MSELoss()

    def penalize_featurecount(self, count):
        return 1e-1 * count

    def create_evalnet(self, D):
        '''
        return torch.nn.Sequential(
            torch.nn.Linear(D, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        '''
        return torch.nn.Sequential(
            torch.nn.Linear(D, 32),

            modules.ResNet(

                modules.ResBlock(
                    block = torch.nn.Sequential(
                        modules.PrototypeClassifier(32, 32),
                        modules.polynomial.Activation(32, n_degree=6),
                        torch.nn.Linear(32, 32)
                    )
                ),

                modules.ResBlock(
                    block = torch.nn.Sequential(
                        modules.PrototypeClassifier(32, 32),
                        modules.polynomial.Activation(32, n_degree=6),
                        torch.nn.Linear(32, 32)
                    )
                )
            ),

            torch.nn.Linear(32, 1)
        )

    def train_evalnet(self, evalnet, X, Y):
        epochs = 300
        lossf = torch.nn.MSELoss()
        optim = torch.optim.Adam(evalnet.parameters())
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 200])

        bar = tqdm.tqdm(range(epochs), ncols=80)
        avg = MovingAverage(momentum=0.95)
        
        for epoch in bar:
            Yh = evalnet(X).squeeze()
            loss = lossf(Yh, Y)
            full_loss = loss + self.grad_penalty(evalnet, X)
            optim.zero_grad()
            full_loss.backward()
            optim.step()
            sched.step()
            avg.update(loss.item())
            bar.set_description("Fitting evalnet: %.3f" % avg.peek())

@main
def main(cycles):
    cycles = int(cycles)
    
    features = Equal4.D
    prog = Equal4(
        gradpenalty_weight = 1e-2,
        init_X = torch.rand(8, features),
        explore = 8,
        exploit = 8,
        table = GlobalOptimizationTable(
            capacity = 4000,
            features = features,
            reduced_size = 3000,
            montecarlo_c = math.sqrt(2)
        ),
        lipo = Lipo(k=1, d=features, a=0, b=1),
        max_retry = 10,
        lr = 1,
        savepath = "equal4.pkl",
        prep_visualization = True
    )

    ground_truth = torch.zeros(features).byte()
    ground_truth[:Equal4.TRUE_D] = 1

    try:
        for i in range(cycles):
            print(" === Epoch %d ===" % i)
            prog.step()
            X, Y = prog.publish_XY()
            top = prog.discretize_featuremask(X[0])
            print("Top feature selection:", top.numpy(), sep="\n")
            if (top == ground_truth).all() and input("Stop? [y/n] ") == "y":
                break
    except KeyboardInterrupt:
        pass
    finally:
        X, Y = prog.publish_XY()

        best_n = 3
        x = (X[:best_n] > NeuralGlobalOptimizer.SELECTION).numpy()
        print(" === Top %d feature selections === " % best_n, x, sep="\n")
        print(" >>> Number of retraining operations: %d" % prog.count_network_retrains())
        
        data_loss, test_loss, feature_counts = prog.get_losses()

        import matplotlib
        matplotlib.use("agg")
        from matplotlib import pyplot
        fig, axes = pyplot.subplots(nrows=3, sharex=True, figsize=(10, 8))

        axes[0].plot(data_loss, ".-")
        axes[0].set_ylabel("Training loss")

        axes[1].plot(test_loss, ".-")
        axes[1].set_ylabel("Validation loss")

        axes[2].plot(feature_counts, ".-")
        axes[2].set_ylabel("Feature count")

        axes[-1].set_xlabel("Evaluations")

        pyplot.savefig("equal4.png")
