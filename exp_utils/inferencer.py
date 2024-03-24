from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from . import JeongganProcessor
from .train_utils import Tokenizer
from .model_zoo import TransformerOMR

class Inferencer:
  def __init__(self, 
               vocab_txt_fn='model/synth_only_240313_002_tokenizer.txt', 
               model_weights='model/synth_only_240313_002_best.pt', 
               device='cuda'):
    self.tokenizer = Tokenizer(vocab_txt_fn=vocab_txt_fn)
    self.model = TransformerOMR(128, vocab_size=len(self.tokenizer.vocab))
    self.model.load_state_dict(torch.load(model_weights, map_location='cpu')['model'])
    self.model.eval()
    self.model.to(device)
    self.transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Grayscale(num_output_channels=1),
      transforms.Lambda(lambda x: 1-x),
      transforms.Resize((160, 140), antialias=True),
      ])
    self.remove_border = JeongganProcessor.remove_borders
    self.device = device
  
  def __call__(self, img):
    if isinstance(img, str) or isinstance(img, Path):
      img = cv2.imread(img)
    
    if isinstance(img, list):
      assert all([isinstance(single_img, np.ndarray) for single_img in img])
      img = [self.transform(self.remove_border(single_img)) for single_img in img]
      img = torch.stack(img, dim=0)
      pred, conf = self.model.inference(img.to(self.device), batch=True, return_confidence=True)
      pred = [self.tokenizer.decode(single_pred) for single_pred in pred]
    else:
      img = self.remove_border(img)
      img = self.transform(img)
      img = img.unsqueeze(0)
      pred = self.model.inference(img.to(self.device), batch=False)
      pred = self.tokenizer.decode(pred)
      conf = None
    return pred, conf
  
