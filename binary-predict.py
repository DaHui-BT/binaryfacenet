import torch
from facenet_pytorch import MTCNN
from model import FaceNet
from PIL import Image
import os

from tools.preprocess import preprocess_image

def extract_face_embedding(img1_path: str, img2_path: str, mtcnn: MTCNN, model: FaceNet):
  img1 = preprocess_image(img1_path)
  img2 = preprocess_image(img2_path)
  img1_cropped = mtcnn(img1).to(device=model.device)
  img2_cropped = mtcnn(img2).to(device=model.device)
  assert img1_cropped is not None and img2_cropped is not None, 'not face has been detected!'
    
  with torch.no_grad():
    embedding = model(img1_cropped.unsqueeze(0), img2_cropped.unsqueeze(0))
  return embedding


def verify(img1_path: str, img2_path: str, mtcnn: MTCNN, model: FaceNet, threshold: int = 0.6):
  embedding = extract_face_embedding(img1_path, img2_path, mtcnn, model)
  
  return embedding.item() > threshold, embedding.item()


if __name__ == '__main__':
  weight = 'checkpoint/model.pt'
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
  model = FaceNet(device=device).eval()

  if os.path.isfile(weight):
    model.load_state_dict(torch.load(weight))
    
  img1_path = 'imgs/Aaron_Eckhart_0001.jpg'
  img2_path = 'imgs/Aaron_Sorkin_0001.jpg'
  # img1_path = 'imgs/Aaron_Sorkin_0001.jpg'
  # img2_path = 'imgs/Aaron_Sorkin_0002.jpg'
  is_same, similarity_score = verify(img1_path, img2_path, mtcnn, model)

  print(f'Are the images of the same person? {is_same}')
  print(f'Similarity Score: {similarity_score}')
