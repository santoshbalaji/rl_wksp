import gymnasium as gym
import numpy as np
import pickle
import gzip

# setting up cartpole environment
env = gym.make("CartPole-v1")

# hyperparameters
learning_rate = 0.001 # learning rate for policy updates
gamma = 0.995         # discount factor
num_episodes = 20000   # total number of episodes

# discretize the state space into bins for each feature
n_bins = [50, 50, 50, 50]  # number of bins for each state variable
state_bins = [
    np.linspace(-4.8, 4.8, n_bins[0] - 1),     # cart position
    np.linspace(-4, 4, n_bins[1] - 1),         # cart velocity
    np.linspace(-0.418, 0.418, n_bins[2] - 1), # pole angle
    np.linspace(-4, 4, n_bins[3] - 1)          # pole angular velocity
]

# initialize policy table as a dictionary
policy = dict()


def discretize_state(state):
    """
      Discretize the continuous state into a tuple of bins.
    """
    state_discrete = tuple(int(np.digitize(s, bins)) for s, bins in zip(state, state_bins))
    return state_discrete


def initialize_state():
    """
      Initialize policy entries for a new discrete state.
    """
    for i in range(0, 50):
        for j in range(0, 50):
            for k in range(0, 50):
                for l in range(0, 50):
                    # initialize with equal probability for both actions (0: left, 1: right)
                    state = (i, j, k, l)
                    policy[state] = {0: 0.5, 1: 0.5}


def select_action(state):
    """
      Select an action based on the policy probabilities.
    """
    action_probabilities = policy[state]
    action = np.random.choice([0, 1], p=[action_probabilities[0], action_probabilities[1]])
    return action


def compute_returns(rewards, gamma):
    """
      Compute the cumulative discounted returns for each step in an episode.
    """
    returns = list()
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


def update_policy(episode_states, episode_actions, episode_returns):
    """
      Update the policy table using the policy gradient approach.
    """
    for state, action, G in zip(episode_states, episode_actions, episode_returns):
        other_action = 1 - action  # The action we didn't take
        # increase probability for actions that lead to high returns
        policy[state][action] += learning_rate * G * (1 - policy[state][action])
        # decrease probability for the other action
        policy[state][other_action] -= learning_rate * G * policy[state][other_action]
        # ensure probabilities remain valid
        policy[state][action] = max(0, min(1, policy[state][action]))
        policy[state][other_action] = 1 - policy[state][action]


def run_training():
    initialize_state()

    for episode in range(num_episodes):
        state = discretize_state(env.reset()[0])

        episode_states = list()
        episode_actions = list()
        episode_rewards = list()

        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state)

            # store the state, action, and reward
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

        # compute returns for the episode
        episode_returns = compute_returns(episode_rewards, gamma)

        # update policy based on the episode
        update_policy(episode_states, episode_actions, episode_returns)

        # logging for tracking progress
        if episode % 100 == 0:
            total_reward = sum(episode_rewards)
            print(f"Episode {episode}, Total Reward: {total_reward}")

    with gzip.open('policy_grad_wk.pickle.gz', 'wb') as handle:
        pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_model():
    global policy
    with gzip.open('policy_grad_wk.pickle.gz', 'rb') as handle:
        policy = pickle.load(handle)

    env = gym.make("CartPole-v1", render_mode='human')
    # run test episodes with trained models
    for episode in range(5):
        state = env.reset()[0]
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            state = discretize_state(state)
            action = select_action(state)

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
    run_training()
    run_model()

# available states
# cart position (how far it is left or right from the center)
# cart velocity (speed and direction of the cart)
# pole angle (the angle of the pole relative to base)
# pole angular velocity (how quickly is falling or returning upright)

# available actions
# left move 0 (move the agent to the left)
# right move 1 (move the agent to the right)