import gym
from skeleton import Agent

env = gym.make("CartPole-v1")
observation_dims = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = Agent(
    observation_dims=observation_dims,
    n_actions=n_actions,
    alpha=0.001,
    gamma=0.9,
    reuse=False
)

for episode in range(1000):

    observation = env.reset().tolist()
    observations = []
    rewards = []

    while True:

        action = agent.choose_action([observation])[0]
        new_observation, reward, done, _ = env.step(action)

        observations.append(observation)
        rewards.append(reward)

        observation = new_observation

        if (done):
            break
    
    print(sum(rewards))
    agent.learn(observations, rewards)

observation = env.reset().tolist()

while True:

    action = agent.choose_action([observation])[1]
    new_observation, reward, done, _ = env.step(action)
    env.render()

    observation = new_observation
    if (done):
        break