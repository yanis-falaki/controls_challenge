from environment import Environment
from stable_baselines3 import PPO
import gymnasium as gym

env = Environment(custom_datapath='./data/00001.csv')
env.reset()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

#episodes = 10
#for ep in range(episodes):
#    obs = env.reset
#    done = False
#    while not done:
#        obs, reward, done, info = env.step(env.action_space.sample())