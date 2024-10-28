import gymnasium as gym

env = gym.make(
    "Blackjack-v1", 
    render_mode="human")

observation, info = env.reset(seed=42)
for i in range(0, 10):
    action = env.action_space.sample()
    
    print(action)

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
