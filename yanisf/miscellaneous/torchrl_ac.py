import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, SafeModule, ActorValueOperator, NormalParamExtractor

class MLPFeatureExtractor(nn.Module):
    def __init__(self, in_features, num_cells):
        super(MLPFeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells),
            nn.Tanh()
        )

    def forward(self, current_state, time):
        print(current_state.shape)
        x = torch.cat([current_state, time], dim=-1)
        return self.net(x)
    

class ActorCritic(ActorValueOperator):
    def __init__(self, in_features, num_actions, low, high, num_cells, in_keys=["current_state", "time"]):
        backbone = MLPFeatureExtractor(in_features, num_cells)
        td_module_hidden = SafeModule(
            module=backbone,
            in_keys=in_keys,
            out_keys=["hidden"]
        )

        actor_head = nn.Sequential(nn.Linear(num_cells, 2*num_actions), NormalParamExtractor())
        td_module_actor = TensorDictModule(
            module=actor_head,
            in_keys=["hidden"],
            out_keys=["loc", "scale"]
        )
        td_module_action = ProbabilisticActor(
            module=td_module_actor,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": low,
                "high": high
            },
            return_log_prob=True,
        )

        critic_head = nn.Linear(num_cells, 1)
        td_module_critic = ValueOperator(
            module=critic_head,
            in_keys=["hidden"]
        )

        super().__init__(td_module_hidden, td_module_action, td_module_critic)