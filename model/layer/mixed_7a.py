import torch
from torch import nn, Tensor

from . import BasicConv2d


class Mixed_7a(nn.Module):

  def __init__(self):
    '''
      Inception block
      [basic_conv2d + basic_conv2d] + [basic_conv2d + basic_conv2d] + 
        [basic_conv2d + basic_conv2d + basic_conv2d] + max_pool2d + concatenation
    '''
    
    super().__init__()

    self.branch0 = nn.Sequential(
      BasicConv2d(896, 256, kernel_size=1, stride=1),
      BasicConv2d(256, 384, kernel_size=3, stride=2)
    )

    self.branch1 = nn.Sequential(
      BasicConv2d(896, 256, kernel_size=1, stride=1),
      BasicConv2d(256, 256, kernel_size=3, stride=2)
    )

    self.branch2 = nn.Sequential(
      BasicConv2d(896, 256, kernel_size=1, stride=1),
      BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
      BasicConv2d(256, 256, kernel_size=3, stride=2)
    )

    self.branch3 = nn.MaxPool2d(3, stride=2)

  def forward(self, x: Tensor) -> Tensor:
    x0 = self.branch0(x)
    x1 = self.branch1(x)
    x2 = self.branch2(x)
    x3 = self.branch3(x)
    out = torch.cat((x0, x1, x2, x3), 1)
    return out

