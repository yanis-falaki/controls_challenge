
import torch
import torch.nn as nn
from tensordict.nn import TensorDictSequential
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ParallelEnv
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchrl.envs.utils import set_exploration_type, ExplorationType

from miscellaneous.tiny_sim_wrapper import TinySimWrapper
from miscellaneous.torchrl_ac import ActorCritic
from miscellaneous.attention_ac import ActorCriticWithAttention

import multiprocessing
import math


class FrameLimitedTrainer():
    def __init__(self, device, env, actor_critic: ActorCritic, frame_limit=200,
                 total_frames_per_env=100_000, frames_per_batch_per_env=1000, minibatch_size=-1, num_epochs=10,
                 lr=1e-3, gamma=0.9, gae_lambda=0.9, clip_eps=0.2, entropy_eps=1e-4, critic_coeff=1.0,):
        
        self.device = device
        self.env = env
        self.actor_critic = actor_critic

        # Detect number of enviornments
        if isinstance(env, ParallelEnv):
            self.num_envs = env.batch_size[0]
        else:
            self.num_envs = 1

        self.total_frames = total_frames_per_env * self.num_envs
        self.frames_per_batch = frames_per_batch_per_env * self.num_envs
        self.num_epochs = num_epochs
        self.episodes_per_batch = frames_per_batch_per_env / frame_limit
        if minibatch_size == -1:
            self.minibatch_size = self.frames_per_batch//10
        else:
            self.minibatch_size = minibatch_size

        self.advantage_module = GAE(gamma=gamma, lmbda=gae_lambda, value_network=actor_critic.get_value_operator(), average_gae=True)
        self.loss_module = ClipPPOLoss(
            actor_network=actor_critic.get_policy_operator(),
            critic_network=actor_critic.get_value_operator(),
            clip_epsilon=clip_eps,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            critic_coef=critic_coeff,
            loss_critic_type="smooth_l1"
        )
        self.optimizer = Adam(self.loss_module.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.total_frames // self.frames_per_batch)

        self.collector = SyncDataCollector(
            env,
            actor_critic.get_policy_operator(),
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=device
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.frames_per_batch, device="auto"),
            sampler=SamplerWithoutReplacement()
        )

    @torch.no_grad()
    @set_exploration_type(ExplorationType.DETERMINISTIC)
    def save(self, file_name="model.pt2"):
        policy = TensorDictSequential(self.actor_critic.get_policy_operator(), selected_out_keys=["action"]).requires_grad_(False).eval()

        if isinstance(self.env, ParallelEnv):
            observation = env.fake_tensordict()["observation"][0]
        else:
            observation = env.fake_tensordict()["observation"]

        model_export = torch.export.export(policy, args=(), kwargs={"observation": observation})

        torch.export.save(model_export, f"./saved_models/{file_name}")
        
        policy.requires_grad_(True).train()

    @torch.no_grad()
    def compute_avg_episode_returns(self, tensordict_data: TensorDict):
        rewards = tensordict_data["next"]["reward"].squeeze(-1)  # [num_envs, total_frames]
        dones = tensordict_data["next"]["done"].squeeze(-1)      # [num_envs, total_frames]

        all_episode_returns = []
        current_returns = torch.zeros(self.num_envs, device=self.device)
        
        for t in range(dones.shape[-1]):
            reward = rewards[:, t]
            done = dones[:, t]
            current_returns += reward
            if done.any():
                all_episode_returns.append(current_returns)
                current_returns = torch.zeros(self.num_envs, device=self.device)
        
        if all_episode_returns:
            return torch.tensor(all_episode_returns).mean()
        else:
            return None


    def train(self, save_model=False):
        best_avg_batch_rewards = -torch.inf
        self.actor_critic.train()

        for i, tensordict_data in enumerate(self.collector):
            for _ in range(self.num_epochs):
                # Reshape tensordict as a single batch dim is needed for attention to work
                tensordict_data = tensordict_data.reshape(-1, *tensordict_data.shape[2:])

                # Calculate Advantage, modifies tensordict_data inplace
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                #self.replay_buffer.extend(data_view.cpu())
                self.replay_buffer.extend(data_view)

                for _ in range(self.frames_per_batch // self.minibatch_size):
                    subdata = self.replay_buffer.sample(self.minibatch_size)
                    
                    #loss_vals = self.loss_module(subdata.to(device))
                    loss_vals = self.loss_module(subdata)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    if torch.isnan(loss_vals["loss_objective"]) or torch.isnan(loss_vals["loss_critic"]) or torch.isnan(loss_vals["loss_entropy"]):
                        print("NAN LOSS DETECTED")

                    self.optimizer.zero_grad()
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), 1.0)
                    self.optimizer.step()

            self.scheduler.step()

            avg_batch_reward = tensordict_data["next"]["reward"].mean()
            if save_model and avg_batch_reward > best_avg_batch_rewards:
                best_avg_batch_rewards = avg_batch_reward
                self.save()

            print(f"Step {i*self.frames_per_batch}, Episode {i*self.episodes_per_batch}, Average Cost: {avg_batch_reward:.2f}, critic_loss {loss_vals["loss_critic"]}")


def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if grad_norm > 1000:
                print(f"Exploding gradient in {name}: {grad_norm}")
            elif grad_norm < 1e-6:
                print(f"Vanishing gradient in {name}: {grad_norm}")

if __name__ == "__main__":
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")

    num_envs = 2
    env = ParallelEnv(num_envs,
        lambda: TinySimWrapper(
            model_path="/home/yanisf/Documents/coding/controls_challenge/models/tinyphysics.onnx",
            data_directory_path="/home/yanisf/Documents/coding/controls_challenge/data"),
        device=device)

    in_features = env.observation_spec["current_state"].shape[-1] + 1
    num_actions = env.action_spec.shape[-1]
    low = env.action_spec_unbatched.space.low
    high = env.action_spec_unbatched.space.high

    #ac = ActorCritic(in_features, num_actions, low, high, 256, in_keys=["current_state", "time"]).to(device)
    ac = ActorCriticWithAttention(num_actions, low, high).to(device)

    trainer = FrameLimitedTrainer(device, env, ac, total_frames_per_env=200000, frames_per_batch_per_env=400, clip_eps=0.2, frame_limit=400)

    trainer.train()