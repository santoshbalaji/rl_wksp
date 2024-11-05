import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# setting up cartpole environment
env = gym.make('CartPole-v1')

class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(ActorCritic, self).__init__()

        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        # actor network
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # for discrete actions
        )

        # critic network
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
    # hyperparameters
    # training_counter = 0
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    learning_rate = 0.01
    gamma = 0.99

    # initialize model, optimizer, and loss function
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for episode in range(2000):
        state = env.reset()
        terminated = False
        truncated = False
        rewards = []
        log_probs = []
        values = []
        episode_reward = 0

        state = state[0]
        while not terminated and not truncated:
            state = torch.FloatTensor(state)
            action_probs, value = model(state)
            
            # sample action from the probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # step environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            episode_reward += reward
            
            state = next_state
        
        # compute advantages
        values = [v.item() for v in values]
        advantages = compute_advantage(rewards, values, gamma)

        # compute loss and update model
        actor_loss = -(torch.stack(log_probs) * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}")

        # early stopping if solved
        # if episode_reward >= 200 and training_counter < 10:
        #     training_counter = training_counter + 1
        #     print(f"trained: {training_counter}")
        # elif episode_reward >= 200 and training_counter >= 10:
        #     print(f"solved in {episode} episodes!")
        #     break

    torch.save(model.state_dict(), 'actor_critic_nn_wk.pth')


def run_with_trained_model():
    env = gym.make("CartPole-v1", render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    test_model = ActorCritic(state_dim, action_dim)

    test_model.load_state_dict(torch.load('actor_critic_nn_wk_bk.pth', weights_only=False))
    test_model.eval()  # set the model to evaluation mode

    # run test episodes
    for episode in range(5):  # run 5 test episodes
        state = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        state = state[0]
        while not terminated and not truncated:
            state = torch.FloatTensor(state)  # convert state to tensor
            action_probs, _ = test_model(state)

            # choose action with the highest probability
            action = torch.argmax(action_probs).item()

            # take the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # update state
            state = next_state

            # render the environment
            env.render()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()


if __name__ == '__main__':
    # run_training()
    run_with_trained_model()

# available states
# cart position (how far it is left or right from the center)
# cart velocity (speed and direction of the cart)
# pole angle (the angle of the pole relative to base)
# pole angular velocity (how quickly is falling or returning upright)

# available actions
# left move 0 (move the agent to the left)
# right move 1 (move the agent to the right)
