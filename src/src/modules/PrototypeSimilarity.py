import torch, math, numpy, matplotlib

matplotlib.use("agg")
from matplotlib import pyplot

from .CosineSimilarity import CosineSimilarity

DEFAULT = CosineSimilarity(dim=2)

class PrototypeSimilarity(torch.nn.Module):

    def __init__(self, features, classes, similarity=DEFAULT):
        super().__init__()
        self.C = classes
        self.D = features
        self.weight = torch.nn.Parameter(torch.zeros(1, classes, features))
        self.similarity = similarity
        self.visualizing = 0
        self.stored_visuals = []
        self._axes = None
        self.reset_parameters()

    def forward(self, X):
        '''

        Input:
            X - torch Tensor of size (N, D, *), input features.

        Output:
            P - torch Tensor of size (N, C, *), prototype classifications.
        
        '''
        #X = X.unsqueeze(1)
        #e = len(X.shape) - len(self.weight.shape)
        #P = self.weight.view(*self.weight.shape, *([1]*e))
        output = torch.nn.functional.tanh(X)#self.similarity(X, P)
        if self.visualizing > 0:
            self.store_visuals(output)
        return output

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def set_visualization_count(self, count):
        self.visualizing = count

    def store_visuals(self, output):
        output = output[:,:self.visualizing].clone().detach()
        
        if len(output.shape) == 4:
            output = output.permute(0, 2, 3, 1).contiguous().view(-1, output.size(1))

        assert len(output.shape) == 2
        
        self.stored_visuals.append(output.cpu().numpy())

    def visualize(self, title, figsize):
        if self.visualizing > 0:
            if self.stored_visuals:
                self.do_visualization(numpy.concatenate(self.stored_visuals, axis=0), title, figsize)
            self.visualizing = 0
            self.stored_visuals = []
            return self._axes

    def do_visualization(self, visuals, title, figsize):
        k = visuals.shape[1]
            
        if self._axes is None:
           _, self._axes = pyplot.subplots(nrows=2, ncols=k, sharex=True, sharey="row", figsize=figsize)

        axes = self._axes[0,:]
        for i in range(k):
            plot = axes[i]
            plot.set_xlim([-1, 1])
            plot.hist(visuals[:,i], bins=100)

        axes[0].set_ylabel(title)
