import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, CONTEXT_LENGTH
from controllers import modified_pid
from functools import partial

class Discretizer():
    def __init__(self, min, max, num_bins):
        self.min = min
        self.max = max
        self.num_bins = num_bins
        

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # Input:
        # Current_Lataccel, Target_Lataccel, roadRoll, aEgo, vEgo

        self.fc1 = nn.Linear(in_features=5, out_features=64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.fc5(x)
        x = (F.sigmoid(x)*4) - 2
        return x