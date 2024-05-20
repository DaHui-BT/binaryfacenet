import torch
from torch import nn, Tensor

from . import BasicConv2d


class Block17(nn.Module):
  
  def __init__(self, scale: int = 1.0) -> None:
    '''
      Inception block with residual connection
      basic_conv2d + [basic_conv2d + basic_conv2d + basic_conv2d] + 
        concatenation + conv2d + residual + relu
        
      Parameters:
      - scale: scaling ratio
    '''
    
    super().__init__()
    self.scale = scale

    self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

    self.branch1 = nn.Sequential(
      BasicConv2d(896, 128, kernel_size=1, stride=1),
      BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
      BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))
    )

    self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
    self.relu = nn.ReLU(inplace=False)

  def forward(self, x: Tensor) -> Tensor:
    x0 = self.branch0(x)
    x1 = self.branch1(x)
    out = torch.cat((x0, x1), 1)
    out = self.conv2d(out)
    out = out * self.scale + x
    out = self.relu(out)
    return out
