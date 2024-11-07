#!/usr/bin/env python3
import argparse
import time
import gymnasium as gym
from ros2_gym_env.ros2_gym_env import RlManipEnv
from stable_baselines3 import PPO, SAC


def parse_arguments():
    """
    Parses command-line arguments for mode selection and rendering options.
    """
    parser = argparse.ArgumentParser(
        description="Train or Test a PPO model on the RlManipEnv environment."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="Train",  # Options: "Train" or "Test"
        help="Choose 'Train' to train the model or 'Test' to evaluate the model.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,  # Options: "human" for live rendering or "rgb_array" for frame capture
        choices=["human", "rgb_array", None],
        help="Set rendering mode to 'human' for live rendering or 'rgb_array' to capture frames.",
    )
    return parser.parse_args()


def train_model(render_mode):
    """
    Trains the PPO model on the RlManipEnv environment.
    """
    # Create the environment for training
    env = RlManipEnv(
        render_mode=render_mode,        # Options: "human" or None
        env_mode="Train",               # Options: "Train" or "Test"
        robot_model="ur5e",             # Options: "ur5e" or "aubo_i5"
        gripper_model=None,             # Options: "AG95" or None
        task="reach",                   # Options: "reach", "reach_box_static", "reach_box_dynamic"
        max_steps=500,
        # initial_pose=[0, 0, 1.5707, 0, 1.5707, 0], # For Aubo_i5
        target_pose=[-0.7, 0.0, 0.18, 0.0, 0.0, 0.0, 1.0],
        reward_mode=1                   #Options: 1-Position 2-Position+Orientation  3-Position+Orientation+Jerk
    )

    observation, info = env.reset(seed=42)

    # Define and train the model
    model = PPO("MlpPolicy", env, verbose=1)
    total_timesteps = 1_00_000  # Adjust as needed
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("ppo_ur5e_ros2_reach_position_training_nov7_1_00_000")

    # Close the environment after training
    env.close()


def test_model(render_mode):
    """
    Tests the trained PPO model on the RlManipEnv environment.
    """
    # Load the previously trained model
    model = PPO.load("ppo_ur5e_ros2_reach_training")

    # Create the environment for testing
    env = RlManipEnv(
        render_mode=render_mode,        # Options: "human" or None
        env_mode="Test",                # Options: "Train" or "Test"
        robot_model="ur5e",             # Options: "ur5e" or "aubo_i5"
        gripper_model=None,           # Options: "AG95" or None
        task="reach",                   # Options: "reach", "reach_box_static", "reach_box_dynamic"
        target_pose=[-0.7, 0.0, 0.18, 0.0, 0.0, 0.0, 1.0],
        max_steps=500,
        reward_mode=1                   #Options: 1-Position 2-Position+Orientation  3-Position+Orientation+Jerk
    )

    observation, info = env.reset(seed=42)

    # Run action loop for testing
    for episode in range(100000):  # Specify number of episodes
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # Check for termination or truncation
        if terminated or truncated:
            time.sleep(1)
            status = "TERMINATED" if terminated else "TRUNCATED"
            print(f"{status} ....")
            observation, info = env.reset(seed=42)

    # Close the environment after testing
    env.close()


def main():
    """
    Main function to execute training or testing based on the provided mode.
    """
    args = parse_arguments()

    if args.mode == "Train":
        train_model(args.render_mode)
    elif args.mode == "Test":
        test_model(args.render_mode)
    else:
        print("INVALID MODE: Please use 'Train' or 'Test'.")


if __name__ == "__main__":
    main()
