import torch
from torch import nn, Tensor

from . import BasicConv2d

class Block35(nn.Module):

  def __init__(self, scale: int = 1.0):
    '''
      Inception block with residual connection
      basic_conv2d + [basic_conv2d + basic_conv2d] +
        [basic_conv2d + basic_conv2d + basic_conv2d] +
        concatenation + conv2d + residual + relu
        
      Parameters:
      - scale: scaling ratio
    '''
    
    super().__init__()
    self.scale = scale

    self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

    self.branch1 = nn.Sequential(
      BasicConv2d(256, 32, kernel_size=1, stride=1),
      BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
    )

    self.branch2 = nn.Sequential(
      BasicConv2d(256, 32, kernel_size=1, stride=1),
      BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
      BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
    )

    self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
    self.relu = nn.ReLU(inplace=False)

  def forward(self, x: Tensor) -> Tensor:
    x0 = self.branch0(x)
    x1 = self.branch1(x)
    x2 = self.branch2(x)
    out = torch.cat((x0, x1, x2), 1)
    out = self.conv2d(out)
    out = out * self.scale + x
    out = self.relu(out)
    return out
