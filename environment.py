import numpy as np
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE, CONTROL_START_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER, CONTEXT_LENGTH
from controllers import CONTROLLERS

class Environment():
    def __init__(self, data_path="./data/00000.csv"):
        model_path = "./models/tinyphysics.onnx"
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
        self.lataccel_costs = []
        self.jerk_costs = []
        self.total_costs = []

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
            Cost (numpy array): 
                lataccel_cost: float
                jerk_cost: float
                total_cost: float
        """

        # Analoguous to sim.sim_control_step
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        self.sim.action_history.append(action)

        self.sim.sim_step(self.sim.step_idx)
        self.sim.step_idx += 1

        done = self.sim.step_idx >= len(self.sim.data)

        if not done:
            state, target = self.sim.get_state_target(self.sim.step_idx)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)

            # For now the cost is just the negative squared difference between actual lataccel and the target at the last timestep
            cost = self.compute_cost()

            return np.array([self.sim.target_lataccel_history[self.sim.step_idx], self.sim.current_lataccel, *self.sim.state_history[self.sim.step_idx]]), cost, done
        else:
            return np.array([0, 0, 0, 0, 0]), [0, 0, 0], done
            
    def compute_cost(self):
        target = self.sim.target_lataccel_history[-1]
        pred = self.sim.current_lataccel_history[-1]

        lataccel_cost = ((target - pred)**2) * 100
        jerk_cost = (((pred - self.sim.current_lataccel_history[-2]) / DEL_T)**2) * 100
        total_cost = (lataccel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost

        return np.array([lataccel_cost, jerk_cost, total_cost])
    
    def get_total_cost(self):
        target = np.array(self.sim.target_lataccel_history)[CONTROL_START_IDX:]
        pred = np.array(self.sim.current_lataccel_history)[CONTROL_START_IDX:]

        lat_accel_cost = np.mean((target - pred)**2) * 100
        jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost

        return np.array([lat_accel_cost, jerk_cost, total_cost])