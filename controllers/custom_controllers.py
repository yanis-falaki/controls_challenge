import sys
sys.path.append('../')

from . import BaseController
from tinyphysics import CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER

import numpy as np

class Controller(BaseController):
  """
  A controller that returns target_lataccel
  """
  def __init__(self):
    super().__init__()
    self.actual_lataccels = []
    self.target_lataccels = []

  def update_history(self, target_lataccel, current_lataccel):
    self.actual_lataccels.append(current_lataccel)
    self.target_lataccels.append(target_lataccel)

  def calculate_costs(self):
    target = np.array(self.target_lataccels)[CONTROL_START_IDX-CONTEXT_LENGTH:COST_END_IDX-CONTEXT_LENGTH]
    pred = np.array(self.actual_lataccels)[CONTROL_START_IDX-CONTEXT_LENGTH:COST_END_IDX-CONTEXT_LENGTH]

    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.update_history(target_lataccel, current_lataccel)
    return 0