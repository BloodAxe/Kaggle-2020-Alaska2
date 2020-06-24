from timm.models.layers import Swish
from torch import nn

__all__ = ["WeightNormClassifier"]

from torch.nn.utils import weight_norm


class WeightNormClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout):
        super().__init__()
        layers = [
            nn.Dropout(dropout, inplace=False),
            weight_norm(nn.Linear(in_dim, hidden_dim), dim=None),
            Swish(),
            weight_norm(nn.Linear(hidden_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
