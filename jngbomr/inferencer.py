from pathlib import Path
from omegaconf import OmegaConf

import cv2
import numpy as np
import torch
from torchvision import transforms

from .jeonggan_utils import JeongganProcessor
from .vocab_utils import Tokenizer
from .model_zoo import TransformerOMR


INFERENCER_DEFAULT_KWARGS = dict(
  vocab_txt_fn='checkpoints/best/tokenizer.txt', 
  model_config_path='checkpoints/best/config.yaml',
  model_weights='checkpoints/best/model.pt',
  device='cuda',
)


class Inferencer:
  def __init__(
    self, 
    vocab_txt_fn='checkpoints/best/tokenizer.txt', 
    model_config_path='checkpoints/best/config.yaml',
    model_weights='checkpoints/best/model.pt',
    device='cuda',
  ):
    conf = OmegaConf.load(model_config_path)
    
    self.device = (
      torch.device(device) 
      if torch.cuda.is_available() 
      else torch.device('cpu')
    )
    
    self.tokenizer = Tokenizer(vocab_txt_fn=vocab_txt_fn)
    self.model = TransformerOMR(
      conf.model.dim, 
      len(self.tokenizer.vocab), 
      enc_depth=conf.model.enc_depth, 
      dec_depth=conf.model.dec_depth, 
      num_heads=8, 
      dropout=conf.model.dropout
    )
    
    self.model.load_state_dict(torch.load(model_weights, map_location='cpu')['model'])
    self.model.eval()
    self.model.to(self.device)
    
    self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Grayscale(num_output_channels=1),
      transforms.Lambda(lambda x: 1-x),
      transforms.Resize((160, 140), antialias=True),
    ])
    
    self.remove_border = JeongganProcessor.remove_borders
  
  
  def __call__(self, img):
    if isinstance(img, str) or isinstance(img, Path):
      img = cv2.imread(img)
    
    if isinstance(img, list):
      assert all([
        isinstance(single_img, np.ndarray) 
        for single_img in img
      ]), "If img is a list, all elements must be numpy arrays."
      assert len(img) > 0, "If img is a list, it must not be empty."
      
      img = [ 
        self.transform( self.remove_border(single_img) ) 
        for single_img in img 
      ]
      img = torch.stack(img, dim=0)
      pred, conf = self.model.inference( 
        img.to(self.device), 
        batch=True, 
        return_confidence=True 
      )
      pred = [ 
        self.tokenizer.decode(single_pred) 
        for single_pred in pred 
      ]
    
    else:
      img = self.remove_border(img)
      img = self.transform(img)
      img = img.unsqueeze(0)
      pred = self.model.inference(img.to(self.device), batch=False)
      pred = self.tokenizer.decode(pred)
      conf = None
    
    return pred, conf
