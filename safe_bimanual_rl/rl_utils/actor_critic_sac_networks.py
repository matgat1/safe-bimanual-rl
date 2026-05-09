import torch
from torch import nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """MLP critic network that maps (state, action) to a scalar Q-value."""

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu") / 2)
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu") / 2)
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear") / 2)

    def forward(self, state, action):
        """Concatenate state and action, then compute Q-value through the MLP."""
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        return torch.squeeze(self._h3(features2))


class ActorNetwork(nn.Module):
    """MLP actor network that maps a state to action logits."""

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu") / 2)
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu") / 2)
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear") / 2)

    def forward(self, state):
        """Pass state through the MLP and return action logits."""
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        return self._h3(features2)
