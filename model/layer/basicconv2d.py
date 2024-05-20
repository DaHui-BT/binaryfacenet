from torch import nn, Tensor


class BasicConv2d(nn.Module):

  def __init__(self, in_planes: int, out_planes: int, kernel_size: int,
               stride: int, padding: int = 0) -> None:
    '''
      Basic convolution
      conv2d + batchnorm2d + relu
      
      Parameters:
      - in_planes: the input channel size
      - out_planes: the output channel size
      - kernel_size: the convolution kernel size
      - stride: the step size
      - padding: padding size
    '''
    
    super().__init__()
    self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
    self.relu = nn.ReLU(inplace=False)

  def forward(self, x: Tensor) -> Tensor:
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x
