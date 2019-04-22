import torch, math, numpy

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
        self.reset_parameters()

    def forward(self, X):
        X = X.unsqueeze(1)
        e = len(X.shape) - len(self.weight.shape)
        P = self.weight.view(*self.weight.shape, *([1]*e))
        output = self.similarity(X, P)
        if self.visualizing > 0:
            self.store_visuals(output)
        return output

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def set_visualization_count(self, count):
        self.visualizing = count

    def store_visuals(self, output):
        self.stored_visuals.append(output[:,:self.visualizing].clone().detach().cpu().numpy())

    def visualize(self, title, figsize):
        if self.visualizing > 0:
            if self.stored_visuals:
                self.do_visualization(numpy.concatenate(self.stored_visuals, axis=0), title, figsize)
            self.visualizing = 0
            self.stored_visuals = []

    def do_visualization(self, visuals):
        pass
