from itertools import count
from environment import Environment
from controllers import PIDController
import pandas as pd

pid = PIDController()
env = Environment()
state = env.reset()
data = []

for i in range(5002, 10000): # 20k different csv files
    pid = PIDController()
    env = Environment(data_path=f'./data/{str(i).zfill(5)}.csv')
    state = env.reset()
    for step in count():
        prev_action = 0
        prev_error = pid.prev_error
        prev_derivative = pid.prev_derivative

        action = pid.update(state[0], state[1], [state[2], state[3], state[4]])

        error = state[0] - state[1]
        integral = pid.integral
        derivative = error - prev_error
        prev_action = action

        state, reward, terminated = env.step(action)

        data.append((*state, error, prev_error, integral, derivative, prev_derivative, prev_action, action))

        if terminated:
            break

print(len(data))
df = pd.DataFrame(data, columns=["target_lataccel", "current_lataccel", "vEgo", "aEgo", "roll", "error", "prev_error", "integral", "derivative", "prev_derivative", "prev_action", "steerCommand"])

df.to_csv('pid_controller_data.csv', index=False, mode='a', header=False)