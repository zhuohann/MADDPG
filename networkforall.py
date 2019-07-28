import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size,hidden_in_dim = 256, hidden_out_dim = 128):
        super(Actor, self).__init__()
        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size,hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        h1 = F.relu(self.fc1(self.bn(states)))
        h2 = F.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        return h3
        
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, num_agent, seed=123, fc_units=512, fc_units1=256, fc_units2=256, fc_units3=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.bn = nn.BatchNorm1d(state_size * num_agent)
        self.fc1 = nn.Linear((state_size + action_size) * num_agent, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units1)
        self.fc3 = nn.Linear(fc_units1, fc_units2)
        self.fc4 = nn.Linear(fc_units2, fc_units3)
        self.fc5 = nn.Linear(fc_units3, 1)

        self.reset_parameters()

    def reset_parameters(self):

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        x = torch.cat((self.bn(state), action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return self.fc5(x)