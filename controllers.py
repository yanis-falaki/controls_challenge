class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
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

  def update(self, target_lataccel, current_lataccel, state):

    error = target_lataccel - current_lataccel
    self.integral += error

    # Raw derivative
    raw_derivative = error - self.prev_error

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


CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController
}
