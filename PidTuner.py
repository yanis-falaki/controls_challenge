from environment import Environment
from controllers import PIDController
from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import VerboseCallback

# Define the objective function
def objective(params):
    Kp, Ki, Kd, Kaw, derivative_filter = params
    
    env = Environment()
    state = env.reset()
    controller = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, Kaw=Kaw, derivative_filter=derivative_filter)
    total_cost = 0

    while True:
        params = [state[0], state[1], (state[2], state[3], state[4])]
        action = controller.update(*params)
        next_state, cost, done = env.step(action)
        state = next_state

        if done:
            total_cost = env.get_total_cost()[2]
            break

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
