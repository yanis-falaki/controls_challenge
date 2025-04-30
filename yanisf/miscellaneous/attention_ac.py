import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, ActorValueOperator, NormalParamExtractor

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=99):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe
    

class AttentionFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionFeatureExtractor, self).__init__()
        self.embed_dim = embed_dim
        self.embedder = nn.Linear(5, embed_dim) # current_lataccel, target_lataccel, roll_lataccel, v_ego, a_ego
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim) 

    def forward(self, past_states, current_state, future_plans):
        # Add space for sequence len
        current_state = current_state.unsqueeze(-2)
        # Add current_lataccel as zero to future plans
        future_plans = torch.cat((future_plans, torch.zeros_like(future_plans[..., :1])), dim=-1)

        sequence = torch.cat((past_states, current_state, future_plans), dim=-2)

        embedded = self.embedder(sequence)
        embedded = self.norm1(embedded)

        embedded = self.pos_encoding(embedded)

        attention_output, _ = self.attention(embedded, embedded, embedded)
        attention_output = self.norm2(attention_output)

        current_output = attention_output[..., 50, :] + embedded[..., 50, :]

        return current_output


class ActorCriticWithAttention(ActorValueOperator):
    def __init__(self, num_actions, low, high, embed_dim=64, num_heads=4):
        # Initialize the transformer-based backbone
        backbone = AttentionFeatureExtractor(embed_dim, num_heads)
        td_module_hidden = TensorDictModule(
            module=backbone,
            in_keys=["past_states", "current_state", "future_plans"],
            out_keys=["hidden"]
        )

        # Actor head: maps hidden state to action distribution parameters
        actor_head = nn.Sequential(nn.Linear(embed_dim, 2 * num_actions), NormalParamExtractor())
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

        # Critic head: maps hidden state to value
        critic_head = nn.Linear(embed_dim, 1)
        td_module_critic = ValueOperator(
            module=critic_head,
            in_keys=["hidden"]
        )

        super().__init__(td_module_hidden, td_module_action, td_module_critic)