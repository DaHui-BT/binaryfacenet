import torch
from torch import nn, Tensor

from . import BasicConv2d


class Block8(nn.Module):

  def __init__(self, scale: int = 1.0, noReLU: bool = False):
    '''
      Inception block with residual connection
      basic_conv2d + [basic_conv2d + basic_conv2d + basic_conv2d] + 
        concatenation + conv2d + residual + (relu)
        
      Parameters:
      - scale: scaling ratio
      - noReLU: using relu or not
    '''
    
    super().__init__()

    self.scale = scale
    self.noReLU = noReLU

    self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

    self.branch1 = nn.Sequential(
      BasicConv2d(1792, 192, kernel_size=1, stride=1),
      BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
      BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))
    )

    self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
    if not self.noReLU:
      self.relu = nn.ReLU(inplace=False)

  def forward(self, x: Tensor) -> Tensor:
    x0 = self.branch0(x)
    x1 = self.branch1(x)
    out = torch.cat((x0, x1), 1)
    out = self.conv2d(out)
    out = out * self.scale + x
    if not self.noReLU:
      out = self.relu(out)
    return out
