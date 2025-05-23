import numpy as np
import torch
from typing import Optional, Union, Tuple
import pandas as pd
from hashlib import md5
import os
import random

import sys
sys.path.append("/home/yanisf/Documents/coding/controls_challenge/") # Bandaid fix

from tinyphysics import TinyPhysicsModel,State, FuturePlan
from tinyphysics import (
    ACC_G, DEL_T, CONTEXT_LENGTH, FUTURE_PLAN_STEPS,
    MAX_ACC_DELTA, CONTROL_START_IDX, LAT_ACCEL_COST_MULTIPLIER,
    STEER_RANGE, MAX_ACC_DELTA, LATACCEL_RANGE, COST_END_IDX
)

from torchrl.data import Composite, Unbounded, Bounded
from tensordict import TensorDict
from torchrl.envs import EnvBase

class TinySimWrapper(EnvBase):
    
    def __init__(self,
        model_path: str,
        data_directory_path: str,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: Optional[torch.Size] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(device=device, batch_size=batch_size)

        self.dtype = torch.float32
        self.end_at_idx = COST_END_IDX

        self.sim_model = TinyPhysicsModel(model_path, debug=True)

        self.data_directory_path = data_directory_path
        self.file_data_list = [os.path.join(data_directory_path, f) for f in os.listdir(data_directory_path) if os.path.isfile(os.path.join(data_directory_path, f))]

        self._reset()


        self.observation_spec = Composite(
            past_states=Bounded(
                shape=(49, 5),
                low=torch.tensor([-torch.inf, -torch.inf, -torch.inf, LATACCEL_RANGE[0], LATACCEL_RANGE[0]], 
                                device=device, dtype=self.dtype).repeat(49, 1),
                high=torch.tensor([torch.inf, torch.inf, torch.inf, LATACCEL_RANGE[1], LATACCEL_RANGE[1]], 
                                device=device, dtype=self.dtype).repeat(49, 1),
            ),
            current_state=Bounded(
                shape=(5,),
                low=torch.tensor([-torch.inf, -torch.inf, -torch.inf, LATACCEL_RANGE[0], LATACCEL_RANGE[0]], 
                                device=device, dtype=self.dtype),
                high=torch.tensor([torch.inf, torch.inf, torch.inf, LATACCEL_RANGE[1], LATACCEL_RANGE[1]], 
                                device=device, dtype=self.dtype),
            ),
            future_plans=Bounded(
                shape=(49, 4),
                low=torch.tensor([-torch.inf, -torch.inf, -torch.inf, LATACCEL_RANGE[0]], 
                                device=device, dtype=self.dtype).repeat(49, 1),
                high=torch.tensor([torch.inf, torch.inf, torch.inf, LATACCEL_RANGE[1]], 
                                device=device, dtype=self.dtype).repeat(49, 1),
            ),
            time=Bounded(
                shape=(1,),
                low=torch.tensor(0, device=device, dtype=self.dtype),
                high=torch.tensor(1, device=device, dtype=self.dtype)
            ),
            shape=()
        )

        # Action Spec
        self.action_spec = Composite(
            action=Bounded(
                low=torch.tensor(STEER_RANGE[0], device=device),
                high=torch.tensor(STEER_RANGE[1], device=device),
                shape=(1,),
                dtype=self.dtype,
                device=device
            ),
            shape=()
        )

        # Reward Spec
        self.reward_spec = Composite(
            reward=Bounded(
                shape=(1,),
                high=[0.0],
                low=[-torch.inf],
                device=device,
                dtype=self.dtype
            ),
            shape=()
        )

        # Done Spec
        self.done_spec = Composite(
            done=Bounded(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool,
                device=device
            ),
            shape=()
        )

    def _reset(
        self,
        tensordict: Optional[TensorDict] = None,
        **kwargs
    ) -> TensorDict:
        """Reset the environment and return to the initial state."""
        self.current_data_path = random.choice(self.file_data_list)
        self.data = self.get_data(self.current_data_path)
        self.tiny_reset()
        self.update_states_targets_futureplan()
        
        while(self.step_idx < CONTROL_START_IDX):
            action = self.data['steer_command'].values[self.step_idx]
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            self.action_history.append(action)
            self.sim_step(self.step_idx)
            self.step_idx += 1
            self.update_states_targets_futureplan()

        current_state_tensor = self.create_current_state_tensor()
        past_states_tensor = self.create_past_states_tensor()
        future_plan_tensor = self.create_future_plan_tensor()
        time = self.create_time_tensor()

        tensordict = TensorDict({
            "current_state": current_state_tensor,
            "past_states": past_states_tensor,
            "future_plans": future_plan_tensor,
            "time": time,
        }, batch_size=self.batch_size, device=self.device)

        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Take a step in the environment using the provided action"""
        # Extract action and complete control_step()
        action = tensordict.get("action").item()
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        self.action_history.append(action)

        self.sim_step(self.step_idx)
        reward = self.compute_single_step_cost(self.step_idx)

        self.step_idx += 1
        self.update_states_targets_futureplan()

        current_state_tensor = self.create_current_state_tensor()
        past_states_tensor = self.create_past_states_tensor()
        future_plan_tensor = self.create_future_plan_tensor()
        time = self.create_time_tensor()

        reward_tensor = torch.tensor([reward], dtype=self.dtype, device=self.device)
        if self.step_idx >= self.end_at_idx:
            done_tensor = torch.tensor([True], dtype=torch.bool, device=self.device)
        else:
            done_tensor = torch.tensor([False], dtype=torch.bool, device=self.device)

        out_tensordict = TensorDict({
            "current_state": current_state_tensor,
            "past_states": past_states_tensor,
            "future_plans": future_plan_tensor,
            "time": time,
            "reward": reward_tensor,
            "done": done_tensor
        }, batch_size=self.batch_size, device=self.device)

        return out_tensordict


    def _set_seed(self, seed: Optional[int] = None) -> None:
        """set the seed for the environment"""
        pass

    def close(self) -> None:
        """close the environment"""
        pass

    @property
    def state_spec(self) -> Composite:
        """Return the state specification"""
        return self.observation_spec
    
    def create_current_state_tensor(self):
        current_state = torch.tensor([
            *self.state_history[self.step_idx],
            self.target_lataccel_history[self.step_idx],
            self.current_lataccel,
        ], dtype=self.dtype, device=self.device)
        return current_state 
    
    def create_past_states_tensor(self):
        past_states = torch.stack([
                torch.tensor([state[0] for state in self.state_history[-50:-1]], dtype=self.dtype, device=self.device),
                torch.tensor([state[1] for state in self.state_history[-50:-1]], dtype=self.dtype, device=self.device),
                torch.tensor([state[2] for state in self.state_history[-50:-1]], dtype=self.dtype, device=self.device),
                torch.tensor(self.target_lataccel_history[-50:-1], dtype=self.dtype, device=self.device),
                torch.tensor(self.current_lataccel_history[-50:-1], dtype=self.dtype, device=self.device),
            ], dim=1)
        return past_states

    def create_future_plan_tensor(self):
        future_state = torch.stack([
                torch.tensor(self.futureplan[1], dtype=self.dtype, device=self.device),
                torch.tensor(self.futureplan[2], dtype=self.dtype, device=self.device),
                torch.tensor(self.futureplan[3], dtype=self.dtype, device=self.device),
                torch.tensor(self.futureplan[0], dtype=self.dtype, device=self.device), # target lataccel
            ], dim=1)
        return future_state
    
    def create_time_tensor(self):
        return torch.tensor([(self.step_idx - CONTROL_START_IDX) / (self.end_at_idx - CONTROL_START_IDX)], dtype=self.dtype, device=self.device)
    
    def compute_single_step_cost(self, step_idx):
        lat_accel_cost = ((self.target_lataccel_history[step_idx] - self.current_lataccel_history[step_idx])**2) * 100
        jerk_cost = (((self.current_lataccel_history[step_idx] - self.current_lataccel_history[step_idx-1]) / DEL_T)**2) * 100
        total_cost = -(lat_accel_cost*LAT_ACCEL_COST_MULTIPLIER + jerk_cost)
        return total_cost
    
    def update_states_targets_futureplan(self):
        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        self.futureplan = futureplan
    
    def tiny_reset(self) -> None:
        self.step_idx = CONTEXT_LENGTH
        state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
        self.state_history = [x[0] for x in state_target_futureplans]
        self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
        self.current_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_future = None
        self.current_lataccel = self.current_lataccel_history[-1]
        seed = int(md5(self.current_data_path.encode()).hexdigest(), 16) % 10**4
        np.random.seed(seed)
        return self.step_idx

    def get_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        processed_df = pd.DataFrame({
        'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
        'v_ego': df['vEgo'].values,
        'a_ego': df['aEgo'].values,
        'target_lataccel': df['targetLateralAcceleration'].values,
        'steer_command': -df['steerCommand'].values  # steer commands are logged with left-positive convention but this simulator uses right-positive
        })
        return processed_df
    
    def sim_step(self, step_idx: int) -> None:
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        if step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

        self.current_lataccel_history.append(self.current_lataccel)

    def get_state_target_futureplan(self, step_idx: int) -> Tuple[State, float, FuturePlan]:
        state = self.data.iloc[step_idx]
        return (
            State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
            state['target_lataccel'],
            FuturePlan(
                lataccel=self.data['target_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                roll_lataccel=self.data['roll_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                v_ego=self.data['v_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                a_ego=self.data['a_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist()
            )
        )