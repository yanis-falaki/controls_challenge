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
  

class SimplePIDController(BaseController):
  def __init__(self, k_p=0.044, k_i=0.1, k_d=-0.035):
    self.k_p = k_p
    self.k_i = k_i
    self.k_d = k_d
    self.integral = 0.0
    self.prev_error = 0.0

  def update(self, target_lataccel, current_lataccel, state):
    error = target_lataccel - current_lataccel
    self.integral += error
    derivative = error - self.prev_error

    steer_action = (
      self.k_p * error
      + self.k_i * self.integral
      + self.k_d * derivative
    )

    self.prev_error = error
    return steer_action

  

class PIDController(BaseController):
  """Basic PID controller"""
  # Parameters were found using bayesian optimization in PidTuner.py
  def __init__(self, Kp=0.047111529072798586, Ki=0.12436441468723247, Kd=-0.024868240310312872, Kaw=1.9469823775916133, derivative_filter=0.23845683708265408):
    from tinyphysics import STEER_RANGE

    self.Kp = Kp # Proportional Gain
    self.Ki = Ki # Integral Gain, dt is absorbed as it's constant
    self.Kd = Kd # Derivative Gain
    self.Kaw = Kaw # Anti-Windup Gain
    self.u_bounds = STEER_RANGE # Min and Max Control Outputs (Taken from tinyphysics.py)
    self.derivative_filter = derivative_filter  # Filter coefficient for the derivative term

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
    
  
class FinedTunedPidMLP(BaseController):
    def __init__(self):
        self.integral = 0
        self.prev_error = 0
        self.prev_action = 0
        self.prev_derivative = 0

        self.model = torch.load('./models/finetuned_pid_mlp.pth').to(device)
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


CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'spid': SimplePIDController,
  'pid': PIDController,
  'pid_mlp': PidMLP,
  'finetuned_pid_mlp': FinedTunedPidMLP
}
