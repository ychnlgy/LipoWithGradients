import torch

from .. import model
import src

class Tier_1600(model.BaseModel):

    def make_layers(self, D):
        return [
            src.modules.Reshape(1, D),
            torch.nn.Conv1d(1, 16, 3, padding=1, stride=2),
            src.modules.Transpose(1, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),

            src.modules.Transpose(1, 2),
            torch.nn.Conv1d(1, 16, 3, padding=1, stride=2),
            src.modules.Transpose(1, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),

            src.modules.Reshape(D//2//2),
            torch.nn.Linear(D//2//2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        ]
