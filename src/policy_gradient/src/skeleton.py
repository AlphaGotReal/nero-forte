import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np

# from random import random

class PolicyNetwork(nn.Module):

    def __init__(self, observation_dims, n_actions):

        super(PolicyNetwork, self).__init__()

        leaky_slope = 0.1

        self.network = nn.Sequential(
            nn.Linear(observation_dims, 256, bias=True),
            nn.LeakyReLU(leaky_slope),

            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(leaky_slope),

            nn.Linear(256, n_actions, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, observation):

        return self.network(observation)

class ActionSpace():

    def __init__(self,
        linear_vel_range:tuple,
        linear_vel_buckets:int,
        angular_vel_range:tuple,
        angular_vel_buckets:int):

        self.V = np.linspace(*linear_vel_range, linear_vel_buckets)
        self.W = np.linspace(*angular_vel_range, angular_vel_buckets)

        self.activity:dict = {}

        count = 0 
        for v in self.V:
            for w in self.W:
                self.activity[count] = (v, w)
                count += 1

    def __len__(self):
        return len(self.activity)

    def __getitem__(self, r):
        return self.activity[r]

class Agent():

    def __init__(self,
        observation_dims,
        n_actions,
        alpha,
        gamma,
        reuse=False
    ):
        
        self.gamma = gamma
        self.n_actions = n_actions
        self.observation_dims = observation_dims

        self.model = PolicyNetwork(observation_dims, n_actions)
        if (reuse):
            self.model.load_state_dict(torch.load(reuse))

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.probabilities = []

    def choose_action(self, observation):

        observation = torch.tensor([observation], dtype=torch.float32)
        probabilities = self.model(observation)

        stochastic_action = torch.multinomial(probabilities, 1).item()
        deterministic_action = probabilities.argmax().item()
        self.probabilities.append(probabilities[0, stochastic_action])

        return stochastic_action, deterministic_action, probabilities.detach().numpy().tolist()

    def learn(self, observations, rewards):
        
        observations = torch.tensor(observations, dtype=torch.float32)
        returns = []

        g = 0 
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.append(g)

        returns = torch.tensor(returns, dtype=torch.float32)
        probabilities = torch.stack(self.probabilities)

        loss = (returns * torch.log(probabilities)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probabilities = []

    def save(self, file_name):
        torch.save(self.model.state_dict(), file_name)

# agent = Agent(
#     observation_dims=4,
#     n_actions=2,
#     alpha=0.001,
#     gamma=0.99,
#     reuse=False
# )

# ob = [[random()]*4]*400
# rewards = [1 for r in range(400)]

# for r in range(400):
#     agent.choose_action([ob[r]])

