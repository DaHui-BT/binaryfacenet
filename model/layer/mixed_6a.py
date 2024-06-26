import torch
from torch import nn, Tensor

from . import BasicConv2d


class Mixed_6a(nn.Module):

  def __init__(self):
    '''
      Inception block
      basic_conv2d + [basic_conv2d + basic_conv2d + basic_conv2d] + 
        max_pool2d + concatenation
    '''
    
    super().__init__()

    self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

    self.branch1 = nn.Sequential(
      BasicConv2d(256, 192, kernel_size=1, stride=1),
      BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
      BasicConv2d(192, 256, kernel_size=3, stride=2)
    )

    self.branch2 = nn.MaxPool2d(3, stride=2)

  def forward(self, x: Tensor) -> Tensor:
    x0 = self.branch0(x)
    x1 = self.branch1(x)
    x2 = self.branch2(x)
    out = torch.cat((x0, x1, x2), 1)
    return out
