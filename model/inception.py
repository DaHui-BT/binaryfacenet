from torch import nn, Tensor
from torch.nn import functional as F

from .layer import *

class InceptionResnet(nn.Module):

  def __init__(self, num_classes: int = None, dropout_prob: float = 0.6):
    '''
      Inception Resnet model

      Parameters:
      - num_classes: number of output classes
      - dropout_prob: dropout probability
    '''
    
    super(InceptionResnet, self).__init__()
    self.num_classes = num_classes

    self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
    self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
    self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.maxpool_3a = nn.MaxPool2d(3, stride=2)
    self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
    self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
    self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
    self.repeat_1 = nn.Sequential(
      Block35(scale=0.17),
      Block35(scale=0.17),
      Block35(scale=0.17),
      Block35(scale=0.17),
      Block35(scale=0.17),
    )
    self.mixed_6a = Mixed_6a()
    self.repeat_2 = nn.Sequential(
      Block17(scale=0.10),
      Block17(scale=0.10),
      Block17(scale=0.10),
      Block17(scale=0.10),
      Block17(scale=0.10),
      Block17(scale=0.10),
      Block17(scale=0.10),
      Block17(scale=0.10),
      Block17(scale=0.10),
      Block17(scale=0.10),
    )
    self.mixed_7a = Mixed_7a()
    self.repeat_3 = nn.Sequential(
      Block8(scale=0.20),
      Block8(scale=0.20),
      Block8(scale=0.20),
      Block8(scale=0.20),
      Block8(scale=0.20),
    )
    self.block8 = Block8(noReLU=True)
    self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(dropout_prob)
    self.last_linear = nn.Linear(1792, 512, bias=False)
    self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
    self.logits = nn.Linear(512, 8631)

    if self.num_classes is not None:
      self.logits = nn.Linear(512, self.num_classes)


  def forward(self, x: Tensor) -> Tensor:
    x = self.conv2d_1a(x)
    x = self.conv2d_2a(x)
    x = self.conv2d_2b(x)
    x = self.maxpool_3a(x)
    x = self.conv2d_3b(x)
    x = self.conv2d_4a(x)
    x = self.conv2d_4b(x)
    x = self.repeat_1(x)
    x = self.mixed_6a(x)
    x = self.repeat_2(x)
    x = self.mixed_7a(x)
    x = self.repeat_3(x)
    x = self.block8(x)
    x = self.avgpool_1a(x)
    x = self.dropout(x)
    x = self.last_linear(x.view(x.shape[0], -1)) # flatten + linear
    x = self.last_bn(x)
    
    if self.num_classes:
      x = self.logits(x)
    else:
      x = F.normalize(x, p=2, dim=1)
    return x
