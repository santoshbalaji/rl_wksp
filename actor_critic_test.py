import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time  # For controlling the frame rate

# Define the Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        
        # Actor output
        self.actor = nn.Linear(128, action_size)
        
        # Critic output
        self.critic = nn.Linear(128, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        
        # Actor network outputs probabilities for actions
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic network outputs the state value
        state_value = self.critic(x)
        
        return action_probs, state_value

# Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.99  # Discount factor
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def train_step(self, state, reward, next_state, log_prob, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)
        
        # Get the value estimates
        _, state_value = self.model(state)
        _, next_state_value = self.model(next_state)
        
        # Calculate the Critic target (TD target)
        target_value = reward + (1 - done) * self.gamma * next_state_value
        
        # Calculate the advantage
        advantage = target_value - state_value
        
        # Critic loss: Mean squared error between target value and actual value
        critic_loss = advantage.pow(2)
        
        # Actor loss: -log(probability) * advantage
        actor_loss = -log_prob * advantage.detach()  # Detach to avoid backprop through critic
        
        # Total loss: Actor loss + Critic loss
        total_loss = (actor_loss + critic_loss).mean()
        
        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

# Training loop with visualization
env = gym.make('CartPole-v1')  # Enable human-readable rendering
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = ActorCriticAgent(state_size, action_size)

episodes = 1000
max_timesteps = 200
rewards_per_episode = []  # Store total rewards for each episode

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    
    for t in range(max_timesteps):
        # env.render()  # Display the game
        # time.sleep(0.01)  # Slow down for better visualization
        
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        # Train the agent with the collected data
        agent.train_step(state, reward, next_state, log_prob, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episode: {episode+1}, Score: {total_reward}")
            rewards_per_episode.append(total_reward)  # Store the score
            break
    
    # After training ends, plot the rewards per episode
    # if episode % 50 == 0 and episode > 0:
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(rewards_per_episode, label="Reward")
    #     plt.xlabel("Episode")
    #     plt.ylabel("Total Reward")
    #     plt.title("Actor-Critic Training Progress")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

env.close()

# After all episodes, plot final result
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Actor-Critic Training Progress")
plt.legend()
plt.grid(True)
plt.show()