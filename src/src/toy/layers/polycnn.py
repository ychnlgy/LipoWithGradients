import torch

from .. import model
import src

class Tier_1600(model.BaseModel):

    def make_layers(self, D):
        return [
            src.modules.Reshape(1, D),
            torch.nn.Conv1d(1, 4, 3, padding=1, stride=2),
            src.modules.Transpose(1, 2),
            src.modules.PrototypeClassifier(4, 16),
            src.modules.polynomial.Activation(16, n_degree=4),
            torch.nn.Linear(16, 1),

            src.modules.Transpose(1, 2),
            torch.nn.Conv1d(1, 4, 3, padding=1, stride=2),
            src.modules.Transpose(1, 2),
            src.modules.PrototypeClassifier(4, 16),
            src.modules.polynomial.Activation(16, n_degree=4),
            torch.nn.Linear(16, 1),

            src.modules.Reshape(D//2//2),
            torch.nn.Linear(D//2//2, 16),
            src.modules.PrototypeClassifier(16, 16),
            src.modules.polynomial.Activation(16, n_degree=4),
            torch.nn.Linear(16, 1)
        ]
