from environment import Environment
from sb3_contrib import RecurrentPPO
import gymnasium as gym

env = Environment(custom_datapath='./data/00000.csv')
env.reset()

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=1e7)

#episodes = 10
#for ep in range(episodes):
#    obs = env.reset
#    done = False
#    while not done:
#        obs, reward, done, info = env.step(env.action_space.sample())