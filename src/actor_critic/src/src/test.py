import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np

# Define the Actor-Critic neural network architecture
class ActorCritic(nn.Module):
    def __init__(self, input_size, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.actor_head = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        state_value = self.critic_head(x)
        return action_probs, state_value

# Function to select actions based on the actor's policy
def choose_action(model, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    action_probs, _ = model(state)
    action = torch.multinomial(action_probs, 1).item()
    return action

# Function to train the Actor-Critic model 
def train_actor_critic(model, optimizer, states, actions, advantages, returns):
    states = torch.from_numpy(states).float()
    actions = torch.from_numpy(actions).long()
    advantages = torch.from_numpy(advantages).float()
    returns = torch.from_numpy(returns).float()

    optimizer.zero_grad()
    action_probs, state_values = model(states)

    # Actor loss
    action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
    actor_loss = -torch.sum(action_log_probs * advantages)

    # Critic loss
    critic_loss = F.mse_loss(state_values, returns)

    # Total loss
    total_loss = actor_loss + critic_loss

    total_loss.backward()
    optimizer.step()

# Training loop
def train_cartpole():
    env = gym.make('CartPole-v1')
    input_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    model = ActorCritic(input_size, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_episodes = 1000
    gamma = 0.99  # Discount factor

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_states, episode_actions, episode_rewards = [], [], []

        while not done:
            action = choose_action(model, state)
            next_state, reward, done, _ = env.step(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

        # Calculate returns and advantages
        returns = np.zeros_like(episode_rewards, dtype=np.float32)
        advantages = np.zeros_like(episode_rewards, dtype=np.float32)
        cumulative_return = 0.0
        cumulative_advantage = 0.0

        for t in reversed(range(len(episode_rewards))):
            cumulative_return = episode_rewards[t] + gamma * cumulative_return
            cumulative_advantage = episode_rewards[t] + gamma * cumulative_advantage

            returns[t] = cumulative_return
            advantages[t] = cumulative_advantage

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Train the Actor-Critic model
        train_actor_critic(model, optimizer, np.vstack(episode_states),
                           np.array(episode_actions), advantages, returns)

        # Print the total reward of the episode
        total_reward = sum(episode_rewards)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    train_cartpole()
