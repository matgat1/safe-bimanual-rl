import torch.nn as nn
import torch.nn.functional as F


class ConstraintNetwork(nn.Module):
    """Two-headed network for the learned CBF: outputs (mean, log_std)."""

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._mu = nn.Linear(n_features, output_shape[0])
        self._log_std = nn.Linear(n_features, output_shape[0])

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._mu.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(
            self._log_std.weight, gain=nn.init.calculate_gain("linear")
        )

    def forward(self, state):
        x = F.relu(self._h1(state.float()))
        x = F.relu(self._h2(x))
        return self._mu(x), self._log_std(x)
