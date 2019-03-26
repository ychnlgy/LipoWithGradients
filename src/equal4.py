import torch, tqdm, math
import torch.utils.data

from NeuralGlobalOptimizer import *
from MovingAverage import MovingAverage

from main import main

class Equal4(NeuralGlobalOptimizer):

    D = 30

    def get_dataset(self):
        N = 500
        true_D = 4
        X = torch.rand(N*2, Equal4.D)
        Y = X[:,:true_D].sum(dim=1)
        return X[:N], Y[:N], X[N:], Y[N:]

    def make_model(self, D):
        return torch.nn.Linear(D, 1)

    def train_model(self, model, lossf, X, Y):
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        epochs = 100

        optim = torch.optim.Adam(model.parameters())
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[50])

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
        return 1e-4 * count**2

    def create_evalnet(self, D):
        return torch.nn.Sequential(
            torch.nn.Linear(D, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1)
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
            loss = lossf(Yh, Y) + self.grad_penalty(evalnet, X)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            avg.update(loss.item())
            bar.set_description("Fitting evalnet: %.3f" % avg.peek())

@main
def main(cycles):
    cycles = int(cycles)
    
    features = Equal4.D
    prog = Equal4(
        init_X = torch.rand(8, features),
        explore = 8,
        exploit = 2,
        table = GlobalOptimizationTable(
            capacity = 4000,
            features = features,
            reduced_size = 3000,
            montecarlo_c = math.sqrt(2)
        ),
        lipo = Lipo(k=1, d=features, a=0, b=1),
        max_retry = 10,
        lr = 0.1,
        savepath = "equal4.pkl",
        prep_visualization = True
    )

    for i in range(cycles):
        print(" === Epoch %d ===" % i)
        prog.step()

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

    axes[0].plot(data_loss)
    axes[0].set_ylabel("Training loss")

    axes[1].plot(test_loss)
    axes[1].set_ylabel("Validation loss")

    axes[2].plot(feature_counts)
    axes[2].set_ylabel("Feature count")

    axes[-1].set_xlabel("Evaluations")

    pyplot.savefig("equal4.png")
