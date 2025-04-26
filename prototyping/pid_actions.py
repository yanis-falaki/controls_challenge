import sys
sys.path.append('../')

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, CONTEXT_LENGTH
from controllers import pid
from functools import partial

model = TinyPhysicsModel("../models/tinyphysics.onnx", debug=True)
controller = pid.Controller()

# For each csv
# Generate a run
# Store state and action

# Save npz or csv with state and action
# Optionally compute a cost and store it

def run_rollout_custom(model_path, controller, data_path):
  tinyphysicsmodel = TinyPhysicsModel(model_path, debug=False)
  sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=False)

  for _ in range(CONTEXT_LENGTH, 0):
    # Step
    sim.step()