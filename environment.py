import numpy as np
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE, CONTROL_START_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER
from controllers import CONTROLLERS

class Environment():
    def __init__(self):
        model_path = "./models/tinyphysics.onnx"
        data_path = "./data/00000.csv"
        controller_name = "simple"
        debug = False

        self.tinyphysics_model = TinyPhysicsModel(model_path, debug)
        self.controller = CONTROLLERS[controller_name]()
        self.sim = TinyPhysicsSimulator(self.tinyphysics_model, data_path, self.controller, debug)
        
    def reset(self):
        """
        Initializes the environment

        Returns:
            State (numpy array):
                target_lataccel: float
                current_lataccel: float
                vEgo: float
                aEgo: float
                roll: float
        """
        self.sim.reset()
        self.initial_steps()

        return np.array([self.sim.target_lataccel_history[self.sim.step_idx], self.sim.current_lataccel, *self.sim.state_history[self.sim.step_idx]])
    
    def initial_steps(self):
        while self.sim.step_idx < CONTROL_START_IDX:
            state, target = self.sim.get_state_target(self.sim.step_idx)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)

            # Sim control step
            action = self.sim.data['steer_command'].values[self.sim.step_idx]
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            self.sim.action_history.append(action)

            self.sim.sim_step(self.sim.step_idx)
            self.sim.step_idx += 1
        
        state, target = self.sim.get_state_target(self.sim.step_idx)
        self.sim.state_history.append(state)
        self.sim.target_lataccel_history.append(target)

    def step(self, action: float):
        """
        Takes a step in the environment.

        Args:
            action: float ∈ [-2, 2].

        Returns:
            State (numpy array):
                target_lataccel: float
                current_lataccel: float
                vEgo: float
                aEgo: float
                roll: float
            Reward: float
        """

        # Analoguous to sim.sim_control_step
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        self.sim.action_history.append(action)

        self.sim.sim_step(self.sim.step_idx)
        self.sim.step_idx += 1

        state, target = self.sim.get_state_target(self.sim.step_idx)
        self.sim.state_history.append(state)
        self.sim.target_lataccel_history.append(target)

        # For now the cost is just the negative squared difference between actual lataccel and the target at the last timestep
        cost = -np.square(self.sim.target_lataccel_history[self.sim.step_idx] - self.sim.current_lataccel)

        return np.array([self.sim.target_lataccel_history[self.sim.step_idx], self.sim.current_lataccel, *self.sim.state_history[self.sim.step_idx]]), cost, self.sim.step_idx >= len(self.sim.data)-1
    
    def compute_cost(self, last_lataccel):
        lataccel_cost = np.square(self.sim.current_lataccel - self.sim.target_lataccel_history[self.sim.step_idx]) * 100
        jerk_cost = np.square((self.sim.current_lataccel - last_lataccel) / DEL_T) * 100
        total_cost = (lataccel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost

        return lataccel_cost, jerk_cost, total_cost