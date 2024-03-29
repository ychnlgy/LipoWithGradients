import torch, tqdm, math, numpy
import torch.utils.data

from NeuralGlobalOptimizer import *
from MovingAverage import MovingAverage
import modules

from main import main

class Equal4(NeuralGlobalOptimizer):

    D = 32
    TRUE_D = 4

    def create_dataset(self):
        N = 500
        X = torch.rand(N*2, Equal4.D)
        Y = X[:,:Equal4.TRUE_D].sum(dim=1)#torch.zeros(N*2)
        #for i in range(1, Equal4.TRUE_D+1):
        #    Y += X[:,i-1]*i*0.5
        #print(X[0], Y[0])
        return X[:N], Y[:N], X[N:], Y[N:]

    def get_dataset_path(self):
        return "equal4-data-d%d.pkl" % Equal4.D

    def make_model(self, D):
        return torch.nn.Sequential(
            torch.nn.Linear(D, 1)
        )

    def train_model(self, model, lossf, X, Y):
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        epochs = 150

        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[80, 120])

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

    def penalize_featurecount(self, count, D):
        return self.expected_train_loss * self.featurepenalty_frac * count/D

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
        if not hasattr(self, "_evalnet"):
            self._evalnet = torch.nn.Sequential(
                torch.nn.Linear(D, 32),

                modules.ResNet(

                    modules.ResBlock(
                        block = torch.nn.Sequential(
                            modules.PrototypeClassifier(32, 32),
                            modules.polynomial.Activation(32, n_degree=6),
                            torch.nn.Linear(32, 32)
                            #torch.nn.ReLU(),
                            #torch.nn.Linear(64, 64),
                            #torch.nn.ReLU(),
                            #torch.nn.Linear(64, 64),
                        )
                    ),

                    modules.ResBlock(
                        block = torch.nn.Sequential(
                            modules.PrototypeClassifier(32, 32),
                            modules.polynomial.Activation(32, n_degree=6),
                            torch.nn.Linear(32, 32)
                            #torch.nn.ReLU(),
                            #torch.nn.Linear(64, 64),
                            #torch.nn.ReLU(),
                            #torch.nn.Linear(64, 64),
                        )
                    )
                ),
                torch.nn.Linear(32, 1)
            )
                
        return self._evalnet

    def train_evalnet(self, evalnet, X, Y):
        print("Evaluation network data size: %d" % X.size(0))
        
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        epochs = 200
        lossf = torch.nn.MSELoss()
        optim = torch.optim.SGD(evalnet.parameters(), lr=0.01, momentum=0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[80, 160])

        bar = tqdm.tqdm(range(epochs), ncols=80)
        avg = MovingAverage(momentum=0.95)
        
        for epoch in bar:
            for Xb, Yb in dataloader:
                Xb = self.create_normal(Xb)
                Yb = self.create_normal(Yb)
                Yh = evalnet(Xb).squeeze()

                loss = lossf(Yh, Yb)
                full_loss = loss + self.grad_penalty(evalnet, X, Xb)
                optim.zero_grad()
                full_loss.backward()
                optim.step()
            
            sched.step()
            avg.update(loss.item())
            bar.set_description("Fitting evalnet: %.3f" % avg.peek())

    def create_normal(self, t):
        return torch.normal(t, 0.001)

def score(pred, true):
    assert pred.size() == true.size()
    acc = (pred == true).long().sum().item()/torch.numel(pred)
    p = true == 1
    sens = ((pred == 1) & p).float().sum().item()/p.long().sum().item()
    spec = ((pred == 0) &~p).float().sum().item()/(~p).long().sum().item()
    f1 = (sens*spec)/(sens+spec)*2
    return acc, sens, spec, f1

@main
def main(cycles, features, true_features, best_n=10):

    import matplotlib
    matplotlib.use("agg")
    from matplotlib import pyplot
    fig, axes = pyplot.subplots(nrows=3, sharex=True, figsize=(10, 8))

    Equal4.TRUE_D = int(true_features)
    cycles = int(cycles)
    features = int(features)
    best_n = int(best_n)
    
    Equal4.D = features
    prog = Equal4(
        features = features,
        top_n = 10,
        gradpenalty_weight = 1e-3,
        explore = 1,
        exploit = 4,
        mutation_rate = 1.0/features,
        expected_train_loss = 0.01,
        featurepenalty_frac = 10,
        table = GlobalOptimizationTable(
            capacity = 1000,
            features = features,
            reduced_size = 800,
            montecarlo_c = math.sqrt(2)
        ),
        lipo = Lipo(k=2, d=features, a=0, b=1),
        max_retry = 10,
        lr = 1,
        savepath = "equal4.pkl",
        prep_visualization = True
    )

    ground_truth = torch.zeros(features).byte()
    ground_truth[:Equal4.TRUE_D] = 1

    plots = ([], [], [])
    try:
        for i in range(cycles):
            print(" === Epoch %d ===" % i)
            prog.step()
            X, Y = prog.publish_XY()
            for v, plot in zip(prog.get_losses(), plots):
                plot.append(v)

            top = prog.discretize_featuremask(X[:best_n])
            print("Acc/Sens/Spec/F1: %.3f/%.3f/%.3f/%.3f" % score(top[0], ground_truth))
            print(" --- Top %d feature selections --- " % best_n)
            for i in range(best_n):
                print("%d)" % (i+1), top[i].numpy(), "%.4f" % Y[i], sep="\t")
            
            print(" >>> Number of retraining operations: %d" % prog.count_network_retrains())
            
            data_loss, test_loss, feature_counts = plots

            axes[0].plot(data_loss, ".-")
            axes[0].set_ylabel("Training loss")

            axes[1].plot(test_loss, ".-")
            axes[1].set_ylabel("Validation loss")

            axes[2].plot(feature_counts, ".-")
            axes[2].set_ylabel("Feature count")

            axes[-1].set_xlabel("Evaluations")

            pyplot.savefig("equal4.png")
            [axis.cla() for axis in axes]
            
            if (top[0] == ground_truth).all() and input("Stop? [y/n] ") == "y":
                break
            
    except KeyboardInterrupt:
        pass

    print("\n[EXITED]\n")
        
