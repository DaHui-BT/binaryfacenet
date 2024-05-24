import os
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN


def preprocess_image(path: str) -> Image:
  image = np.array(Image.open(path))

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  equalized_image = cv2.equalizeHist(gray_image)

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  clahe_image = clahe.apply(equalized_image)

  processed_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

  return Image.fromarray(processed_image)


def preprocess_and_save_images(root_dir, output_dir, mtcnn):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  for label_name in tqdm(os.listdir(root_dir), ncols=100):
    label_dir = os.path.join(root_dir, label_name)
    if os.path.isdir(label_dir):
      output_label_dir = os.path.join(output_dir, label_name)
      if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
          
      for image_name in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image_device = torch.from_numpy(np.array(image)).to(device=mtcnn.device)
        
        boxes, _ = mtcnn.detect(image_device)
        if boxes is not None:
          box = boxes[0]
          box = [int(b) for b in box]
          image = image.crop(box)
        
        output_image_path = os.path.join(output_label_dir, image_name)
        image.save(output_image_path)

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  mtcnn = MTCNN(device=device)
  
  preprocess_and_save_images('../dataset/lfw_funneled', '../dataset/processed', mtcnn)
