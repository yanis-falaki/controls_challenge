import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import os

# Custom Dataset class
class CarDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = data[['vEgo', 'aEgo', 'roll', 'targetLateralAcceleration']].values
        self.labels = data['steerCommand'].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

# Function to load and combine data
def load_data_from_csvs(directory):
    combined_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            data = data.dropna(subset=['steerCommand'])
            combined_data.append(data)
    combined_data = pd.concat(combined_data, ignore_index=True)
    return combined_data

# MLPController Model class
class MLPControllerModel(nn.Module):
    def __init__(self):
        super(MLPControllerModel, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x