from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__=="__main__":
    env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, device="cpu")
    model.learn(total_timesteps=25_000)
