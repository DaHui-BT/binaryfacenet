import torch
from torch import Tensor
from torch.utils.data import Dataset
import os
import random
from PIL import Image
import numpy as np


class TripletFaceDataset(Dataset):
  def __init__(self, dataset_dir, transform=None):
    self.dataset_dir = dataset_dir
    self.transform = transform
    self.identities = os.listdir(dataset_dir)
    self.identity_to_images = {identity: os.listdir(os.path.join(dataset_dir, identity)) for identity in self.identities}

  def __len__(self):
    return len(self.identities)

  def __getitem__(self, idx):
    anchor_identity = self.identities[idx]
    anchor_images = self.identity_to_images[anchor_identity]

    anchor_image = random.choice(anchor_images)
    
    positive_image = random.choice(anchor_images)

    negative_identity = random.choice(self.identities)
    while negative_identity == anchor_identity:
      negative_identity = random.choice(self.identities)
    negative_images = self.identity_to_images[negative_identity]
    negative_image = random.choice(negative_images)

    anchor_img_path = os.path.join(self.dataset_dir, anchor_identity, anchor_image)
    positive_img_path = os.path.join(self.dataset_dir, anchor_identity, positive_image)
    negative_img_path = os.path.join(self.dataset_dir, negative_identity, negative_image)
    anchor_img = Image.open(anchor_img_path).convert('RGB')
    positive_img = Image.open(positive_img_path).convert('RGB')
    negative_img = Image.open(negative_img_path).convert('RGB')

    if self.transform:
      anchor_img = self.transform(anchor_img)
      positive_img = self.transform(positive_img)
      negative_img = self.transform(negative_img)

    return anchor_img, positive_img, negative_img


class BinaryFaceDataset(Dataset):
  def __init__(self, train: bool = True, transform = None) -> None:
    super(BinaryFaceDataset, self).__init__()
    self.base_path = 'dataset/processed'
    self.train_pair_path = 'dataset/pairsDevTrain.txt'
    self.test_pair_path = 'dataset/pairsDevTest.txt'
    self.transform = transform
    
    if train: path = self.train_pair_path
    else: path = self.test_pair_path
  
    with open(path) as file:
      texts = file.readlines()
      pair_list = []
      
      for text in texts[1:]:
        text = text.removesuffix('\n')
        pair = text.split('\t')
        if len(pair) == 3:
          pair_list.append([f'{pair[0]}/{pair[0]}_{'0' * (4-len(pair[1]))}{pair[1]}.jpg',
                            f'{pair[0]}/{pair[0]}_{'0' * (4-len(pair[2]))}{pair[2]}.jpg', 1])
        elif len(pair) == 4:
          pair_list.append([f'{pair[0]}/{pair[0]}_{'0' * (4-len(pair[1]))}{pair[1]}.jpg',
                            f'{pair[2]}/{pair[2]}_{'0' * (4-len(pair[3]))}{pair[3]}.jpg', 0])
     
    self.pair_list = pair_list
    
    
  def __getitem__(self, index: int):
    first = os.path.join(self.base_path, self.pair_list[index][0])
    second = os.path.join(self.base_path, self.pair_list[index][1])
    
    first = self.transform(Image.open(first))
    second = self.transform(Image.open(second))
    
    return first, second, self.pair_list[index][2]
  
  def __len__(self):
    return len(self.pair_list)
