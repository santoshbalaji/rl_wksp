import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Create the CartPole environment
env = gymnasium.make("CartPole-v1", render_mode='human')

# Hyperparameters
learning_rate = 0.01
gamma = 0.99  # Discount factor for rewards

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, n_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

# Define the Policy Gradient Agent
class PolicyGradientAgent:
    def __init__(self, n_actions, input_dim):
        self.policy = PolicyNetwork(input_dim, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

    def choose_action(self, state):
        # Convert state to torch tensor and get action probabilities
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = self.policy(state)
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def discount_rewards(self, rewards):
        # Apply the discount factor to rewards
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[i]
            discounted_rewards[i] = cumulative
        return discounted_rewards

    def update_policy(self, states, actions, rewards):
        # Compute the discounted rewards
        discounted_rewards = self.discount_rewards(rewards)
        # Normalize rewards for better stability
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-10)

        # Convert states, actions, and rewards to torch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Calculate loss and perform gradient ascent
        self.optimizer.zero_grad()
        log_probs = torch.log(self.policy(states))
        selected_log_probs = rewards * log_probs[range(len(actions)), actions]
        loss = -selected_log_probs.mean()  # Gradient ascent to maximize reward
        loss.backward()
        self.optimizer.step()

# Training parameters
n_episodes = 500
n_actions = env.action_space.n
input_dim = env.observation_space.shape[0]

# Instantiate the policy gradient agent
agent = PolicyGradientAgent(n_actions, input_dim)

# Training loop
for episode in range(n_episodes):
    state = env.reset()
    states, actions, rewards = [], [], []
    done = False
    total_reward = 0

    state = state[0]
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)

        # Record states, actions, and rewards
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        total_reward += reward

    # Update the policy at the end of the episode
    agent.update_policy(np.array(states), np.array(actions), np.array(rewards))

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()