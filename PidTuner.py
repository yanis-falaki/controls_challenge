from controllers import PIDController
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import VerboseCallback

# Define the objective function
def objective(params):
    Kp, Ki, Kd, Kaw, derivative_filter = params

    tinyphysicsmodel = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    data_path = Path('./data/')
    assert data_path.is_dir(), "data_path should be a directory"

    costs = []
    files = sorted(data_path.iterdir())[:100]
    for data_file in tqdm(files, total=len(files)):
        sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_file), controller=PIDController(Kp, Ki, Kd, Kaw, derivative_filter), debug=False)
        cost = sim.rollout()
        costs.append(cost)

    costs_df = pd.DataFrame(costs)
    total_cost = np.mean(costs_df['total_cost'])

    return total_cost

# Define the search space
space = [
    Real(-2, 2.0, name='Kp'),
    Real(-2, 2.0, name='Ki'),
    Real(-2, 2.0, name='Kd'),
    Real(-2, 2.0, name='Kaw'),
    Real(-2, 2.0, name='derivative_filter')
]

# Perform Bayesian optimization with progress callback
result = gp_minimize(objective, space, n_calls=150, random_state=0, callback=[VerboseCallback(n_total=150)])

# Get the best parameters and cost
best_params = result.x
best_cost = result.fun

print(f"Best Cost: {best_cost}")
print(f"Best Parameters: Kp={best_params[0]}, Ki={best_params[1]}, Kd={best_params[2]}, Kaw={best_params[3]}, derivative_filter={best_params[4]}")