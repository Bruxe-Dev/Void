"""PyTorch LSTM analyst (architecture unchanged from sprint1-data-engine.ipynb)."""

import torch
import torch.nn as nn


class ForexLSTMAnalyst(nn.Module):
    """
    Long Short-Term Memory (LSTM) network matching the architecture
    trained in sprint1-data-engine.ipynb.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 3, num_layers: int = 2):
        super(ForexLSTMAnalyst, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        final_time_step_output = out[:, -1, :]
        return self.fc(final_time_step_output)
