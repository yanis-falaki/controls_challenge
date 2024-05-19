import torch
from torch.distributions import Normal, Categorical
import numpy as np
from PidMLP import PidMLP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    print('STATE: ', state)
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3
  

class PIDController(BaseController):
  """Basic PID controller"""
  def __init__(self):
    from tinyphysics import STEER_RANGE

    self.Kp = 0.2 # Proportional Gain
    self.Ki = 0.03 # Integral Gain, dt is absorbed as it's constant
    self.Kd = 0.01 # Derivative Gain
    self.Kaw = 0.03 # Anti-Windup Gain
    self.u_bounds = STEER_RANGE # Min and Max Control Outputs (Taken from tinyphysics.py)
    self.derivative_filter = 0.6  # Filter coefficient for the derivative term

    self.integral = 0 # Accumulative Integral
    self.filtered_derivative = 0  # Initial filtered derivative value
    self.prev_error = 0 # Error on the last update
    self.u_prev = 0 # Previous control output
    self.prev_derivative = 0

  def update(self, target_lataccel, current_lataccel, state):

    error = target_lataccel - current_lataccel
    self.integral += error

    # Raw derivative
    raw_derivative = error - self.prev_error
    self.prev_derivative = raw_derivative

    # Apply the low-pass filter to the derivative term
    self.filtered_derivative = self.derivative_filter * self.filtered_derivative + (1 - self.derivative_filter) * raw_derivative

    # Computing the control output
    u_calculated = self.Kp * error + self.Ki * self.integral + self.Kd * self.filtered_derivative
    u_actual = max(self.u_bounds[0], min(self.u_bounds[1], u_calculated))

    # Anti-windup adjustment
    if u_calculated != u_actual:
      self.integral += self.Kaw * (u_actual - u_actual)

    self.prev_error = error
    self.u_prev = u_actual

    return u_actual
  

class PidMLP(BaseController):
    def __init__(self):
        self.integral = 0
        self.prev_error = 0
        self.prev_action = 0
        self.prev_derivative = 0

        self.model = torch.load('./models/mlp_pid.pth').to(device)
        self.model.eval()

        action_space_n = 50
        self.action_bins = np.linspace(-2, 2, action_space_n)

    def update(self, target_lataccel, current_lataccel, state):
        error = target_lataccel - current_lataccel
        self.integral += error
        derivative = error - self.prev_error

        action_vector = self.model(torch.tensor([[target_lataccel, current_lataccel, state[0], state[1], state[2], 
                                                 error, self.prev_error, self.integral, derivative, self.prev_derivative, self.prev_action]], dtype=torch.float32, device=device))[0]
        action_idx = torch.argmax(action_vector)
        action = self.action_bins[action_idx]

        self.prev_error = error
        self.prev_derivative = derivative
        self.prev_action = action

        return action

class ActorCriticControllerV1(BaseController):
    def __init__(self):
      self.model = torch.load('./models/actor_critic.pth').to(device)
      self.model.eval()

    def update(self, target_lataccel, current_lataccel, state):
        state_tensor = torch.tensor([[target_lataccel, current_lataccel, state[0], state[1], state[2]]], device=device, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        mean_log_std, state_value = self.model(state_tensor)
        # Split mean and log_std
        mean, log_std = mean_log_std.chunk(2, dim=-1)
        # Exponentiate log to bring back to normal
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        return action.item()
    
class ActorCriticControllerV2(BaseController):
    def __init__(self):
      from tinyphysics import STEER_RANGE
      action_space_n = 15
      self.model = torch.load('./models/actor_critic.pth').to(device)
      self.model.eval()
      self.actions = np.linspace(STEER_RANGE[0], STEER_RANGE[1], action_space_n + 1)

    def update(self, target_lataccel, current_lataccel, state):
        state_tensor = torch.tensor([[target_lataccel, current_lataccel, state[0], state[1], state[2]]], dtype=torch.float32, device=device)
        probs, state_value = self.model(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        return self.actions[action.item()]


CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
  'a2c1': ActorCriticControllerV1,
  'a2c2': ActorCriticControllerV2,
  'pid_mlp': PidMLP
}
