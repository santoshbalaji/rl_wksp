import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Create the CartPole environment
env = gymnasium.make("CartPole-v1", render_mode='human')

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
        return torch.softmax(self.fc2(x), dim=1)


def select_action(state):
    state = torch.from_numpy(state).float()
    action_probs = policy_net(state)
    action = np.random.choice(env.action_space.n, p=action_probs.detach().numpy())
    return action, action_probs[action]



policy_net = PolicyNetwork()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)




# cart position (how far it is left or right from the center)
# cart velocity (speed and direction of the cart)
# pole angle (the angle of the pole relative to base)
# pole angular velocity (how quickly is falling or returning upright)
print(env.observation_space.shape[0])