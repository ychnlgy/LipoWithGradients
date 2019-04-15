import torch

from .. import model
import src

class BaseModel(model.BaseModel):

    def create_single_layer(self, input_size, output_size):
        return torch.nn.Sequential(
            torch.nn.Tanh(),
            src.modules.polynomial.Activation(input_size=input_size, n_degree=4),
            torch.nn.Linear(input_size, output_size)
        )

class BaseResModel(model.BaseResModel):

    def create_single_layer(self, input_size, output_size):
        return BaseModel.create_single_layer(self, input_size, output_size)

# === Standard layer-by-layer networks ===

class Tier_100(BaseModel):
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 5),
            self.new_layer(5, 3)
        ]

class Tier_best(BaseModel):

    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 32),

            src.modules.ResNet(

                src.modules.ResBlock(
                    torch.nn.Sequential(
                        src.modules.PrototypeClassifier(32, 32),
                        src.modules.polynomial.Activation(input_size=32, n_degree=4),
                        torch.nn.Linear(32, 32)
                    )
                ),

                src.modules.ResBlock(
                    torch.nn.Sequential(
                        src.modules.PrototypeClassifier(32, 32),
                        src.modules.polynomial.Activation(input_size=32, n_degree=4),
                        torch.nn.Linear(32, 32)
                    )
                ),


                src.modules.ResBlock(
                    torch.nn.Sequential(
                        src.modules.PrototypeClassifier(32, 32),
                        src.modules.polynomial.Activation(input_size=32, n_degree=4),
                        torch.nn.Linear(32, 32)
                    )
                )

            ),

            torch.nn.Linear(32, 1)
        ]
    
class Tier_best0(BaseResModel):
    
    def get_channel_width(self):
        return 16

class Tier_200(BaseModel):
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 7),
            self.new_layer(7, 4),
            self.new_layer(4, 1)
        ]

class Tier_400(BaseModel):
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 8),
            self.new_layer(8, 8),
            self.new_layer(8, 8)
        ]

class Tier_800(BaseModel):
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 16),
            self.new_layer(16, 10),
            self.new_layer(10, 4)
        ]

class Tier_1600(BaseModel):

    '''

    Tanh:
        torch.nn.Linear(D, 32),
        self.new_layer(32, 32),
        self.new_layer(32, 1)

    PrototypeClassifier:
        torch.nn.Linear(D, 24),
        self.new_layer(24, 16),
        self.new_layer(16, 1)

    '''
    
    def make_layers(self, D):
        return [
            torch.nn.Linear(D, 32),
            self.new_layer(32, 32),
            self.new_layer(32, 1)
        ]

class Tier_12800(BaseModel):
    
    def make_layers(self, D):
        return [
            self.new_layer(D, 64),
            self.new_layer(64, 54),
            torch.nn.Linear(54, 24)
        ]

# === Residual networks ===

class Tier_res1600(BaseResModel):

    '''

    Tanh:
        16

    PrototypeClassifier:
        12
        
    '''

    def get_channel_width(self):
        return 16
    
class Tier_res3200(BaseResModel):

    def get_channel_width(self):
        return 18

class Tier_res6400(BaseResModel):

    def get_channel_width(self):
        return 26

class Tier_res12800(BaseResModel):

    def get_channel_width(self):
        return 38

class Tier_res25600(BaseResModel):

    def get_channel_width(self):
        return 54
