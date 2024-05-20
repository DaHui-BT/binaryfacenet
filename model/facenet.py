import torch
from torch import nn, Tensor
from torch.nn import functional as F

from . import InceptionResnet


class FaceNet(nn.Module):
  
  def __init__(self, num_classes: int = None, dropout_prob: float = 0.6,
               alpha: float = .2, esp: float = .00000001, device: str = None):
    '''
      FaceNet model

      Parameters:
      - weight_path: the model's weight path location
      - num_classes: number of output classes
      - dropout_prob: dropout probability
      - device: the model train or evaluate on which device
    '''
    
    super(FaceNet, self).__init__()
    self.alpha = alpha
    self.esp = esp
    self.device = device
    
    self.extract = InceptionResnet(num_classes, dropout_prob)
    self.linear1 = nn.Linear(in_features=1024, out_features=512)
    self.linear2 = nn.Linear(in_features=512, out_features=128)
    self.linear3 = nn.Linear(in_features=128, out_features=1)
    self.sigmoid = nn.Sigmoid()
    
    # self.extract.requires_grad_(False)
    if device is not None:
      self.to(device=device)
  
  
  def forward(self, *images: Tensor | list[Tensor]) -> Tensor:
    '''
      Parameters:
      - images: if images is one value, then the half should be anchor batch,
                  and the other half should be compared batch.
                when images is two values, the first should be anchor batch,
                  the second should be compared batch
      
      Return: the batch label tensor
    '''
    if len(images) == 1:
      batch_size = int(images[0].shape[0] / 2)
      assert images[0].shape[0] % 2 == 0
      
      embedding = self.extract(images[0])
      anchor_embedding, comparative_embedding = embedding[:batch_size], embedding[batch_size:]
    elif len(images) == 2:
      anchor_embedding = self.extract(images[0])
      comparative_embedding = self.extract(images[1])
    else:
      raise ValueError('images parameter should contain one or two values')
    
    
    embedding1 = torch.cat((anchor_embedding, comparative_embedding), dim=1)
    embedding2 = torch.abs(anchor_embedding - comparative_embedding)
    distance = F.pairwise_distance(anchor_embedding, comparative_embedding, keepdim=True)
    x = torch.divide(embedding1, distance + self.esp) * (1 - self.alpha) + embedding1  * self.alpha
    x = F.normalize(x, p=2, dim=1)
    x = F.leaky_relu(self.linear1(x) + embedding2)
    x = F.leaky_relu(self.linear2(x))
    x = self.linear3(x)
    x = self.sigmoid(x)
    
    return x.squeeze(1)
  