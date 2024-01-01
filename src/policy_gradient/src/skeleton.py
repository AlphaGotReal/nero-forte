import torch
import torch.nn as nn
import torch.optim as optim

from random import random

class PolicyNetwork(nn.Module):

    def __init__(self, observation_dims, n_actions):

        super(PolicyNetwork, self).__init__()

        leaky_slope = 0.1

        self.base_network = nn.Sequential(
            nn.Linear(observation_dims, 256, bias=True),
            nn.LeakyReLU(leaky_slope),

            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(leaky_slope)
        )

        self.mean_network = nn.Sequential(
            nn.Linear(256, n_actions, bias=True),
            nn.Tanh()
        )

        self.std_network = nn.Sequential(
            nn.Linear(256, n_actions, bias=True),
            nn.Softplus()
        )

    def forward(self, observation):

        base_out = self.base_network(observation)
        mean = self.mean_network(base_out)
        std = self.std_network(base_out)

        return mean, std

class Agent():

    def __init__(self,
    observation_dims,
    n_actions,
    alpha,
    gamma,
    reuse=False):
        
        self.gamma = gamma
        self.model = PolicyNetwork(observation_dims, n_actions)

        if (reuse):
            self.model.load_state_dict(torch.load(reuse))

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.action_values = []
        self.means = []
        self.stds = []

    def choose_action(self, observation):

        observation = torch.tensor([observation], dtype=torch.float32)
        mean, std = self.model(observation)
        action = torch.normal(mean, std)

        self.action_values.append(action[0])
        self.means.append(mean[0])
        self.stds.append(std[0])

        return mean[0], std[0], action[0]

    def learn(self, observations, rewards):

        observations = torch.tensor(observations, dtype=torch.float32)
        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.append(g)

        returns = torch.tensor(returns, dtype=torch.float32)

        action_values = torch.stack(self.action_values)
        means = torch.stack(self.means)
        stds = torch.stack(self.stds)

        means, stds = self.model(observations)
        log_probabilities = -(0.5 * ((action_values - means)/stds)**2 + torch.log(stds * 1.414 * 1.772)).sum(dim=1)

        loss = (returns * log_probabilities).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.action_values = []
        self.means = []
        self.stds = []

    def save(self, file_name):
        torch.save(self.model.state_dict(), file_name)
   
# agent = Agent(
#     observation_dims=4,
#     n_actions=2,
#     alpha=0.001,
#     gamma=0.99,
#     reuse=False)

# observations = [[random()]*4]*600
# rewards = [random()]*600

# for ob in observations:
#     agent.choose_action(ob)


