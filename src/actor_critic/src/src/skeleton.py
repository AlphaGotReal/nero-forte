import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):

    def __init__(self, observation_size, n_actions):

        super(Actor, self).__init__()
        leaky_slope = 0.1

        self.base = nn.Sequential(
            nn.Linear(observation_size, 256, bias=True),
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

        base_out = self.base(observation)
        mean = self.mean_network(base_out)
        std = self.std_network(base_out)
        return mean, std
    
class Critic(nn.Module):

    def __init__(self, observation_size, n_actions):

        super(Critic, self).__init__()
        leaky_slope = 0.1

        self.observation_network = nn.Sequential(
            nn.Linear(observation_size, 128, bias=True),
            nn.LeakyReLU(leaky_slope),
        )

        self.action_network = nn.Sequential(
            nn.Linear(n_actions, 128, bias=True),
            nn.LeakyReLU(leaky_slope)
        )

        self.combined_network = nn.Sequential(
            nn.Linear(256, 64, bias=True),
            nn.LeakyReLU(leaky_slope),

            nn.Linear(64, 1, bias=True)
        )

    def forward(self, observation, action_value):

        observation = self.observation_network(observation)
        action_value = self.action_network(action_value)
        combined = torch.cat([observation, action_value], dim=1)

        return self.combined_network(combined)

class Agent():

    def __init__(self, 
        observation_size, 
        n_actions, 
        alpha, 
        beta,
        gamma,
        reuse=False
    ):
        
        self.observation_size = observation_size
        self.n_actions = n_actions

        self.gamma = gamma

        self.actor = Actor(observation_size=observation_size, n_actions=n_actions)
        self.critic = Critic(observation_size=observation_size, n_actions=n_actions)
        if (reuse):
            self.actor.load_state_dict(torch.load(f"actor_{reuse}.pth"))
            self.critic.load_state_dict(torch.load(f"critic_{reuse}.pth"))

        self.optimize_actor = optim.SGD(self.actor.parameters(), lr=alpha)
        self.optimize_critic = optim.SGD(self.critic.parameters(), lr=beta)

        self.critic_loss_criteria = nn.MSELoss()

    def choose_action(self, observation):

        #observation = torch.tensor(observation, dtype=torch.float32)
        mean, std = self.actor(observation)
        return process_action(mean, std)
    
    def backprop(self, curr_observations, next_observations, action_values, rewards, terminals):
        
        #curr_observations = torch.tensor(curr_observations, dtype=torch.float32)
        #next_observations = torch.tensor(next_observations, dtype=torch.float32)
        #action_values = torch.tensor(action_values, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminals = torch.tensor(terminals, dtype=torch.float32)

        action_means, action_stds = self.actor(curr_observations)
        curr_q_values = self.critic(curr_observations, action_values)
        with torch.no_grad():
            next_q_values = self.critic(next_observations, self.choose_action(next_observations))
            target_q_values = rewards + self.gamma * next_q_values * (1 - terminals)

        critic_loss = self.critic_loss_criteria(curr_q_values, target_q_values)

        self.optimize_critic.zero_grad()
        critic_loss.backward()
        self.optimize_critic.step()

        advantage = target_q_values - curr_q_values.detach()
        log_probability = (-torch.log(2 * torch.pi * action_stds ** 2) - ((action_values - action_means)/action_stds) ** 2) * 0.5
        actor_loss = -(advantage * log_probability).mean()
        
        self.optimize_actor.zero_grad()
        actor_loss.backward()
        self.optimize_actor.step()

        return actor_loss, critic_loss

    def save(self, name):
        torch.save(self.actor.state_dict(), f"actor_{name}.pth")
        torch.save(self.critic.state_dict(), f"critic_{name}.pth")

def process_action(action_mean, action_std):
    actions = torch.normal(action_mean, action_std)
    return actions

agent = Agent(4, 2, 0.01, 0.01, 0.99)
observations = torch.rand((600, 4), dtype=torch.float32)
next_observations = torch.rand((600, 4), dtype=torch.float32)
actions_values = agent.choose_action(observations)
rewards = [[0.5]] * 600
terminals = [[0]] * 600
