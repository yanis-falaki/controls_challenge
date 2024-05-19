import torch
from torch import nn
import torch.nn.functional as F

class ActorCriticLSTM(nn.Module):
    def __init__(self, device):
        super(ActorCriticLSTM, self).__init__()
        self.input_size = 5
        self.hidden_size = 64
        self.num_layers = 2

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, device=device)

        # Actor fully connected layers
        self.actor_mean_fc = nn.Linear(self.hidden_size, 1) # Predicting mean
        self.actor_std_fc = nn.Linear(self.hidden_size, 1) # Predicting log s.t.d.

        # Critic fully connected layer
        self.critic_fc = nn.Linear(self.hidden_size, 1)

        self.saved_actions = []
        self.saved_targets = []
        self.saved_action_results = []

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x, (self.hidden, self.cell))
        # Take output at last time step
        out = out[:, -1, :]

        # Actor output
        mean = 2 * torch.tanh(self.actor_mean_fc(out)) # Bounding between -2 and 2
        log_std = F.softplus(self.actor_mean_fc(out)) # Forcing std to be positive

        # Critic output
        state_value = self.critic_fc(out)

        return mean, log_std, state_value
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden and cell states with zeros
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return self.hidden, self.cell
    

class ActorCritic(nn.Module):
    def __init__(self, observation_space_n=5, action_space_n=20):
        super(ActorCritic, self).__init__()
        self.action_space_n = action_space_n

        self.fc1 = nn.Linear(observation_space_n, 164)
        self.fc2 = nn.Linear(164, 164)
        self.fc3 = nn.Linear(164, 128)

        self.actor = nn.Linear(128, action_space_n)

        self.critic = nn.Linear(128, 1) # Single action value

        self.saved_actions = []
        self.rewards = []
        self.integral = 0
        self.prev_error = 0
        self.u_prev = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_prob, state_values
