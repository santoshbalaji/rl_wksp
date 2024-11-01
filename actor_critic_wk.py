import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set up the environment
env = gymnasium.make('CartPole-v1', render_mode='human')

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):

        super(ActorCritic, self).__init__()

        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # for discrete actions
        )

        # Critic network
        self.critic = nn.Linear(128, 1)


    def forward(self, x):
        x = self.common(x)
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value


def compute_advantage(rewards, values, gamma):
    returns = []
    G = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    return returns - torch.tensor(values)


def run_training():
    # Hyperparameters
    training_counter = 0
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    learning_rate = 0.001
    gamma = 0.99

    # Initialize model, optimizer, and loss function
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for episode in range(1000):
        state = env.reset()
        done = False
        rewards = []
        log_probs = []
        values = []
        episode_reward = 0

        state = state[0]
        while not done:
            state = torch.FloatTensor(state)
            action_probs, value = model(state)
            
            # Sample action from the probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Step environment
            next_state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            episode_reward += reward
            
            state = next_state
        
        # Compute advantages
        values = [v.item() for v in values]
        advantages = compute_advantage(rewards, values, gamma)

        # Compute loss and update model
        actor_loss = -(torch.stack(log_probs) * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}")

        # Early stopping if solved
        if episode_reward >= 250 and training_counter < 10:
            training_counter = training_counter + 1
            print(f"trained: {training_counter}")
        elif episode_reward >= 250 and training_counter >= 10:
            print(f"Solved in {episode} episodes!")
            break

    torch.save(model.state_dict(), 'a2cm.pt')


def run_with_trained_model():
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    test_model = ActorCritic(state_dim, action_dim)

    test_model.load_state_dict(torch.load('a2cm.pt', weights_only=False))
    test_model.eval()  # Set the model to evaluation mode

    # Run test episodes
    for episode in range(5):  # Run 5 test episodes
        state = env.reset()
        done = False
        total_reward = 0

        state = state[0]
        while not done:
            state = torch.FloatTensor(state)  # Convert state to tensor
            action_probs, _ = test_model(state)

            # Choose action with the highest probability
            action = torch.argmax(action_probs).item()

            # Take the action in the environment
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Update state
            state = next_state

            # Render the environment (optional)
            env.render()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()


if __name__ == '__main__':
    # run_training()
    # run_with_trained_model()
    state_dim = env.observation_space
    action_dim = env.action_space
    print(state_dim)
    print(action_dim)