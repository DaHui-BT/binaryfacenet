import torch
from torch import nn, Tensor
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import FaceNet
from dataset import BinaryFaceDataset
from tools.earlystopping import EarlyStopping


batch_size = 100
learning_rate = 0.0001
num_epochs = 100
img_shape = (160, 160)
t_max = 5
eta_min = 1e-5
image_dir = 'dataset/processed'
weight_path = './checkpoint/model.pt'
log_dir = './log'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cal_accuracy(x: Tensor, label: Tensor, threshold: float = 0.5):
  predicted_labels = torch.zeros_like(x)
  predicted_labels[x > threshold] = 1
  correct_predictions = (predicted_labels == label).float()
  accuracy = correct_predictions.mean() * 100.0
  return accuracy

def process(model: nn.Module, dataloader: DataLoader, is_train: bool = True) -> float:
  if is_train:
    title = 'train'
    model.train()
  else:
    title = 'evaluate'
    model.eval()
  
  loop = tqdm(dataloader, total=len(dataloader), ncols=100)
  total_loss, total_accuracy = 0.0, 0.0
  for anchor, compare, label in loop:
    anchor, compare, label = (anchor.to(device),compare.to(device),
                              label.to(device, dtype=torch.float))

    if is_train:
      optimizer.zero_grad()
    
    with torch.set_grad_enabled(is_train):
      predict = model(torch.cat([anchor, compare], dim=0))
    
      loss = loss_fn(predict, label)
      if is_train:
        loss.backward()
        optimizer.step()
    
    accuracy = cal_accuracy(predict, label)
    
    total_loss += loss.item()
    total_accuracy += accuracy.item()

    loop.set_description(f'{title} [{epoch+1}/{num_epochs}]')
    loop.set_postfix({'loss': loss.item(), 'accuracy': accuracy.item()})
    torch.cuda.empty_cache()

  if is_train: scheduler.step()

  avg_loss = total_loss / len(dataloader)
  avg_accuracy = total_accuracy / len(dataloader)
  print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}')
  writer.add_scalar(tag=f'{title}/loss', scalar_value=avg_loss, global_step=epoch+1)
  writer.add_scalar(tag=f'{title}/accuracy', scalar_value=avg_accuracy, global_step=epoch+1)
  
  return avg_loss


if __name__ == '__main__':
  transform = transforms.Compose([
      transforms.Resize(img_shape),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  # load the dataset
  train_dataset = BinaryFaceDataset(train=True, transform=transform)
  test_dataset = BinaryFaceDataset(train=False, transform=transform)
  # combine to batch dataset
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  
  model = FaceNet(device=device)
  # load the weight file to this model
  if os.path.isfile(weight_path):
    model.load_state_dict(torch.load(weight_path))

  early_stopping = EarlyStopping()
  loss_fn = nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  writer = SummaryWriter(log_dir)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                   T_max=t_max, eta_min=eta_min)

  for epoch in range(num_epochs):
    avg_loss = process(model, train_loader, is_train=True)
    avg_loss = process(model, test_loader, is_train=False)
    # save the model's weight
    early_stopping(avg_loss, model.state_dict(), path=weight_path)
    # if the patient time is end, then jump the loop
    if early_stopping.early_stop:
      break
