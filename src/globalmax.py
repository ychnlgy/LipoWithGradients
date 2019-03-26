import torch, math

from GlobalOptimizer import *

class Example1D(GlobalOptimizer):

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
        return -torch.sin(x)/((x/4-math.pi*2)**2+1)

    def fit_evalnet(self, X, Y):
        '''

        Input:
            X - torch Tensor of shape (N, D), feature selection masks.
            Y - torch Tensor of shape (N), scores for each selection.

        Output:
            evalnet - torch.nn.Module, Lipschitz network that maps
                feature selection to predicted score.

        '''
        return self.evaluate

    @staticmethod
    def main():

        import tqdm, numpy
        from matplotlib import pyplot
        pyplot.figure(figsize=(10, 8))
        
        features = 1
        a = -10
        b = 40
        
        example1d = Example1D(
            init_X = torch.rand(1, 1),
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
            lr = 1,
            savepath = "globalmax.pkl"
        )

        x = torch.linspace(a, b, 1000)
        y = example1d.evaluate(x)

        pyplot.plot(x.numpy(), y.numpy(), "r-", label="True function", alpha=0.6)
        
        for i in range(5):
            example1d.step()

        X, Y = example1d.publish_XY()
        X = X.squeeze().numpy()
        Y = Y.squeeze().numpy()

        pyplot.hist(
            X,
            bins=numpy.linspace(a, b, 100),
            density=True,
            alpha=0.2,
            label="Normalized histogram of Monte Carlo choices"
        )

        n = 50
        x = X[:n]
        y = Y[:n]
        pyplot.plot(x, y, "b.", label="Top %d sampled points" % n, alpha=0.2)

        pyplot.plot(x[:1], y[:1], "bx", label="Candidate global maximum")

        pyplot.title("%d evaluations" % example1d.count_evals())
        pyplot.legend()
        pyplot.savefig("example-1d.png")

if __name__ == "__main__":
    Example1D.main()
    
