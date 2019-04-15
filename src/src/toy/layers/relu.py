import torch

from .. import model

class BaseModel(model.BaseModel):

    def create_single_layer(self, input_size, output_size):
        return torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(input_size, output_size)
        )

class BaseResModel(model.BaseResModel):

    def create_single_layer(self, input_size, output_size):
        return BaseModel.create_single_layer(self, input_size, output_size)

# === Standard layer-by-layer networks ===

class Tier_1layer(BaseModel):

    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 1)
        ]

class Tier_100(BaseModel):
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 32),
            self.new_layer(32, 2)
        ]

class Tier_best(BaseResModel):
    
    def get_channel_width(self):
        return 128

class Tier_200(BaseModel):
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 12),
            self.new_layer(12, 8),
            self.new_layer(8, 2)
        ]

class Tier_400(BaseModel):
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 32),
            self.new_layer(32, 16),
            self.new_layer(16, 2)
        ]

class Tier_800(BaseModel):
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 28),
            self.new_layer(28, 20),
            self.new_layer(20, 2)
        ]

class Tier_1600(BaseModel):
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 32),
            self.new_layer(32, 32),
            self.new_layer(32, 10)
        ]

class Tier_12800(BaseModel):

    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 112),
            self.new_layer(112, 84),
            self.new_layer(84, 32)
        ]

# === Residual networks ===

class Tier_res1600(BaseResModel):

    def get_channel_width(self):
        return 18

class Tier_res3200(BaseResModel):

    def get_channel_width(self):
        return 26

class Tier_res6400(BaseResModel):

    def get_channel_width(self):
        return 38

class Tier_res12800(BaseResModel):

    def get_channel_width(self):
        return 54

class Tier_res25600(BaseResModel):

    def get_channel_width(self):
        return 74
