import torch
from torch import nn, tensor, Tensor

__all__ = ["SelfAttention"]

from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """
    https://github.com/sdoria/SimpleSelfAttention
    """

    def __init__(self, channels: int, sym=False):
        super().__init__()

        self.conv = spectral_norm(nn.Conv2d(channels, channels, 1, bias=False))

        self.gamma = nn.Parameter(tensor([0.0]), True)

        self.sym = sym
        self.n_in = channels

    def forward(self, x: Tensor) -> Tensor:
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t()) / 2
            self.conv.weight = c.view(self.n_in, self.n_in, 1)

        size = x.size()
        x = x.view(*size[:2], -1)  # (C,N)

        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))

        convx = self.conv(x)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())  # (C,N) * (N,C) = (C,C)   => O(NC^2)

        o = torch.bmm(xxT, convx)  # (C,C) * (C,N) = (C,N)   => O(NC^2)

        o = self.gamma * o + x

        return o.view(*size).contiguous()
