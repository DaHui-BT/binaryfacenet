import torch
from torch import nn, Tensor
from facenet_pytorch import MTCNN
from model import InceptionResnet
from PIL import Image
import os

from tools.preprocess import preprocess_image

def extract_face_embedding(img_path: str, mtcnn: MTCNN, model: InceptionResnet):
    img = preprocess_image(img_path)
    img_cropped = mtcnn(img)
    assert img_cropped is not None, 'not face has been detected!'
    
    with torch.no_grad():
        embedding = model(img_cropped.unsqueeze(0))
    return embedding


def cosine_similarity(embedding1: Tensor, embedding2: Tensor):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(embedding1, embedding2).item()


def verify(img1_path: str, img2_path: str, mtcnn: MTCNN, model: InceptionResnet, threshold: int = 0.6):
    embedding1 = extract_face_embedding(img1_path, mtcnn, model)
    embedding2 = extract_face_embedding(img2_path, mtcnn, model)
    
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity > threshold, similarity


if __name__ == '__main__':
  weight = 'checkpoint/inception-model.pt'
  mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40)
  model = InceptionResnet().eval()

  if os.path.isfile(weight):
    model.load_state_dict(torch.load(weight), strict=True)
    
  # img1_path = 'imgs/Aaron_Eckhart_0001.jpg'
  # img2_path = 'imgs/Aaron_Sorkin_0001.jpg'
  img1_path = 'imgs/Aaron_Sorkin_0001.jpg'
  img2_path = 'imgs/Aaron_Sorkin_0002.jpg'
  is_same, similarity_score = verify(img1_path, img2_path, mtcnn, model)

  print(f'Are the images of the same person? {is_same}')
  print(f'Similarity Score: {similarity_score}')
