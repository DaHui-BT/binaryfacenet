import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = .0):
      '''
      Arguments:
        - patience: the loss value above the lowest value times
        - delta: patient for the loss value can be float
      '''
      self.patience = patience
      self.counter = 0
      self.best_score = None
      self.early_stop = False
      self.val_loss_min = np.Inf
      self.delta = delta

    def __call__(self, val_loss: float, state_dict: dict, path: str = './model.pt'):
      score = -val_loss

      if self.best_score is None:
        self.best_score = score
        self.save_checkpoint(val_loss, state_dict, path)
      elif score < self.best_score + self.delta:
        self.counter += 1
        print(f'patient times: [{self.counter} / {self.patience}]')
        if self.counter >= self.patience:
          self.early_stop = True
      else:
        self.best_score = score
        self.save_checkpoint(val_loss, state_dict, path)
        self.counter = 0

    def save_checkpoint(self, val_loss: float, state_dict: dict, path: str):
      '''
        Saves model when validation loss decrease.
      '''
      torch.save(state_dict, path)
      self.val_loss_min = val_loss