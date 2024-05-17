import torch
from torch import nn

class ActorCriticLSTM(nn.Module):
    def __init__(self):
        super(ActorCriticLSTM, self).__init__()
        self.input_size = 5
        self.hidden_size = 64
        self.num_layers = 2

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        # Actor fully connected layer
        self.actor_fc = nn.Linear(self.hidden_size, 2) # Predicting mean and log s.t.d

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
        mean_and_log_std = self.actor_fc(out)

        # Critic output
        state_value = self.critic_fc(out)

        return mean_and_log_std, state_value
    
    def init_hidden(self, batch_size):
        # Initialize hidden and cell states with zeros
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)
        self.cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)
        return self.hidden, self.cell
