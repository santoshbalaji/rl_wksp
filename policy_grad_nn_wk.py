import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# setting up cartpole environment
env = gymnasium.make("CartPole-v1", render_mode='human')

# hyperparameters
learning_rate = 0.01
gamma = 0.99
num_of_episodes = 1000


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

policy_net = PolicyNetwork()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

def check_cuda():
    print(f"Check CUDA availability: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    cuda_id = torch.cuda.current_device()
    print(f"CUDA device id: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(cuda_id)}")


def select_action(state):
    state = torch.from_numpy(state).float()
    # provides probability for each possible actions given its states
    action_probs = policy_net(state)
    # chooses random action based on provided probablity
    # available actions [0, 1]
    action = np.random.choice(env.action_space.n, p=action_probs.detach().numpy())
    return action, action_probs[action]


def compute_returns(rewards, gamma):
    returns = list()
    G = 0
    # computing cummilative discounted reward from the start of this episode
    for reward in reversed(rewards):
        # the rewards obtained for this episode are added with discount factor where 
        # the earliest reward with get the highest portion
        G = reward + (gamma * G)
        returns.insert(0, G)
    returns = torch.tensor(returns)
    return (returns - returns.mean()) / (returns.std() + 1e-8)


def run_training():
    for episode in range(num_of_episodes):
        state = env.reset()[0]
        log_probs = list()
        rewards = list()

        terminated = False
        truncated = False
        while not terminated and not truncated:
            # selecting a new action given its current state
            action, log_prob = select_action(state)
            # running step for this episode with above selected action
            state, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(torch.log(log_prob))
            rewards.append(reward)
        returns = compute_returns(rewards, gamma)

        policy_loss = list()

        # computing policy loss
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        # backpropagation to adjust the weights based on policy loss
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    
    torch.save(policy_net.state_dict(), 'policy_grad_nn_wk.pth')


def run_model():
    test_model = PolicyNetwork()

    test_model.load_state_dict(torch.load('policy_grad_nn_wk_bk.pth', weights_only=False))
    test_model.eval()

    # run test episodes with trained models
    for episode in range(5):
        state = env.reset()[0]
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            state = torch.FloatTensor(state)  # convert state to tensor
            action_probs = test_model(state)

            # choose action with the highest probability
            print(action_probs)
            action = torch.argmax(action_probs).item()

            # take the action in the environment
            next_state, reward, terminated, truncated, _  = env.step(action)
            total_reward += reward

            # update state
            state = next_state

            # render the environment
            env.render()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()


if __name__ == '__main__':
    # check_cuda()
    # run_training()
    run_model()

# available states
# cart position (how far it is left or right from the center)
# cart velocity (speed and direction of the cart)
# pole angle (the angle of the pole relative to base)
# pole angular velocity (how quickly is falling or returning upright)

# available actions
# left move 0 (move the agent to the left)
# right move 1 (move the agent to the right)