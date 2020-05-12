import torch
from torch import nn, tensor, Tensor


class SelfAttention(nn.Module):
    """
    https://github.com/sdoria/SimpleSelfAttention
    """
    def __init__(self, channels: int, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(tensor([0.0])
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        size = x.size()

        x = x.view(*size[:2], -1)  # (C,N)

        convx = self.conv(x)
        xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())
        o = torch.bmm(xxT, convx)
        o = self.gamma * o + x

        return o.reshape(*size)
