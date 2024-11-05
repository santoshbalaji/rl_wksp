import gymnasium as gym
from ros2_gym_env.ros2_gym_env import UR5eGrpEnvROS2
from stable_baselines3 import PPO
import argparse
import time

# Set up argument parser for train or test mode
parser = argparse.ArgumentParser(description="Train or Test PPO model on UR5eGrpEnvROS2 environment.")
parser.add_argument("--mode", type=str, default="Train", help="Set to 'Train' to train the model or 'Test' to test the model")
parser.add_argument("--render_mode", type=str, default=None, choices=["human", "rgb_array", None],
                    help="Set rendering mode to 'human' for live rendering or 'rgb_array' to capture frames.")
args = parser.parse_args()

##############################################################################################################################
################                                             TRAIN                              ##############################
##############################################################################################################################

if args.mode == "Train":
    # Create the environment for training with the specified render mode
    env = UR5eGrpEnvROS2(render_mode=args.render_mode)  # Pass the render mode argument

    # Reset the environment with a specific seed for reproducibility
    observation, info = env.reset(seed=42)

    # Define the model with PPO algorithm
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model for a specified number of timesteps
    total_timesteps = 1000000
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model to the current working directory 
    model.save("ppo_ur5e_ros2_3")

    # Close the environment after training
    env.close()

##############################################################################################################################
################                                             TEST                              ###############################
##############################################################################################################################

elif args.mode == "Test":
    # Load the trained model
    model = PPO.load("ppo_ur5e_ros2_3")

    # Re-create the environment after loading the model with specified render mode
    env = UR5eGrpEnvROS2(render_mode=args.render_mode)
    observation, info = env.reset(seed=42)

    # Run the action loop for testing the model
    for episode in range(100):  # Run for 100 episodes
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Print loop entry and handle termination/truncation
        if terminated or truncated:
            time.sleep(1)
            print("TERMINATED ...." if terminated else "TRUNCATED ....")
            observation, info = env.reset(seed=42)
            break  # Exit the episode loop on termination/truncation

    # Close the environment after testing
    env.close()
else:
    print("INVALID MODE: Please use 'Train' or 'Test'.")

