import torch, math

from GlobalOptimizer import *

class Example2D(GlobalOptimizer):

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
        return 1/(1+(x[0]-3)**2+(x[1]+2)**2)

    def fit_evalnet(self, X, Y):
        '''

        Input:
            X - torch Tensor of shape (N, D), feature selection masks.
            Y - torch Tensor of shape (N), scores for each selection.

        Output:
            evalnet - torch.nn.Module, Lipschitz network that maps
                feature selection to predicted score.

        '''
        return lambda X: self.evaluate(X.transpose(0, 1))

    @staticmethod
    def main():

        import tqdm, numpy
        import matplotlib
        matplotlib.use("agg")
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        ax = Axes3D(pyplot.figure(figsize=(10, 8)))

        features = 2

        a = -10
        b = 10
        n = 1000

        example2d = Example2D(
            init_X = torch.rand(8, features),
            explore = 8,
            exploit = 2,
            table = GlobalOptimizationTable(
                capacity = 2000,
                features = features,
                reduced_size = 1000,
                montecarlo_c = math.sqrt(2)
            ),
            lipo = Lipo(k=1, d=features, a=a, b=b),
            max_retry = 10,
            lr = 0.2,
            savepath = "globalmax.pkl"
        )

        xb = torch.linspace(a, b, n)
        x1 = xb.view(-1, 1).repeat(1, n)
        x2 = xb.view(1, -1).repeat(n, 1)
        x = torch.stack([x1, x2], dim=0)
        y = example2d.evaluate(x)
        ax.plot_surface(x1.numpy(), x2.numpy(), y.numpy(), cmap="hot", label="True function", alpha=0.6)

        for i in tqdm.tqdm(range(200)):
            example2d.step()

        print("Number of function evaluations: %d" % example2d.count_evals())

        X, Y = example2d.publish_XY()
        X = X.squeeze().numpy()
        Y = Y.squeeze().numpy()

        n = 200
        x = X[:n]
        y = Y[:n]
        ax.scatter(x[:,0], x[:,1], y, c="b", label="Top %d sampled points" % n)

        pyplot.title("Global maximum search using gradient-boosted LIPO")
        pyplot.savefig("simple-2d.png")

if __name__ == "__main__":
    Example2D.main()
