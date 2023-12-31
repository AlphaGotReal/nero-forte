import torch
import torch.nn as nn
import torch.optim as optim 

class Model(nn.Module):

    def __init__(self, observation_dims, n_actions):

        super(Model, self).__init__()

        leaky_slope = 0.1

        self.base_network = nn.Sequential(
            nn.Linear(observation_dims, 256, bias=True),
            nn.LeakyReLU(leaky_slope)
        )

    

