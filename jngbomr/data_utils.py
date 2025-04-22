import random
from pathlib import Path
from operator import itemgetter

import pandas as pd
import cv2

import torch
import torch.nn as nn
from torchvision import transforms

from .jeonggan_utils import JeongganProcessor, JeongganSynthesizer
from .vocab_utils import Tokenizer


def pad_collate(raw_batch):
  """
  raw batch is a list of tuples (img, annotation)
  img is torch tensor with shape (1, H, W)
  pad to same width by adding 0s to the left and right
  pad to same height by adding 0s to the top and bottom
  """

  # find max width and height
  max_width = max([img.shape[2] for img, _ in raw_batch])
  max_height = max([img.shape[1] for img, _ in raw_batch])

  img_batch = torch.zeros((len(raw_batch), 1, max_height, max_width))
  for i, (img, _) in enumerate(raw_batch):
    h, w = img.shape[1], img.shape[2]
    left_pad = (max_width - w) // 2
    top_pad = (max_height - h) // 2
    img_batch[i, :, top_pad:top_pad+h, left_pad:left_pad+w] = img

  
  max_token_length = max([len(label) for _, label in raw_batch])
  label_batch = torch.zeros((len(raw_batch), max_token_length), dtype=torch.long)
  for i, (_, label) in enumerate(raw_batch):
    label_batch[i, :len(label)] = torch.tensor(label)
  
  return img_batch, label_batch[:, :-1], label_batch[:, 1:]



class RandomBoundaryDrop:
  def __init__(self, amount=3) -> None:
    self.amount = amount
  
  def __call__(self, img):
    rand_num = random.random()
    boundary_amount = random.randint(1, self.amount)
    if rand_num < 0.25:
      return img[:, :-boundary_amount] 
    elif rand_num < 0.5:
      return img[:, boundary_amount:]
    elif rand_num < 0.75:
      return img[:-boundary_amount, :]
    else:
      return img[boundary_amount:, :]



class Dataset:
  def __init__(self, csv_path, img_path_dict, synth_config={}, is_valid=False) -> None:
    self.df = pd.read_csv(csv_path)
    self.img_path_dict = img_path_dict
    self.jng_synth = self._make_jng_synth(img_path_dict)
    self.is_valid = is_valid
    self.synth_config = synth_config
    self.need_random = not is_valid

    if self.is_valid:
      self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Grayscale(num_output_channels=1),
      transforms.Lambda(lambda x: 1-x),
      transforms.Resize((160, 140), antialias=True),
      ])
    else:
      self.transform = transforms.Compose([
      RandomBoundaryDrop(4),
      transforms.ToTensor(),
      transforms.Grayscale(num_output_channels=1),
      transforms.Lambda(lambda x: 1-x),
      transforms.RandomResizedCrop((160, 140), scale=(0.85, 1,0), antialias=True),
      ])
    self.tokenizer = self._make_tokenizer()


  @staticmethod
  def _make_jng_synth(img_path_dict):
    return JeongganSynthesizer(img_path_dict)


  def _make_tokenizer(self):
    return Tokenizer(self.df['label'].values.tolist())
  

  def get_item_by_idx(self, idx):
    row = self.df.iloc[idx]
    annotation, width, height = itemgetter('label', 'width', 'height')(row)
    img = None
    
    if self.synth_config:
      while True:
        try:
          img = self.jng_synth.generate_image_by_label(annotation, width, height, **self.synth_config)
          break
        except:
          pass
    
    else:
      while True:
        try:
          img = self.jng_synth.generate_image_by_label(annotation, width, height, char_variant=self.need_random, apply_noise=self.need_random, random_symbols=self.need_random, layout_elements=self.need_random)
          break
        except:
          pass
    
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    return img, annotation
  

  def __len__(self):
    return len(self.df)
  

  def __getitem__(self, idx):
    img, annotation = self.get_item_by_idx(idx)
    img = self.transform(img)
    return img, self.tokenizer(annotation)



class LabelStudioDataset(Dataset):
  def __init__(self, csv_path, img_path_dir, is_valid=False, remove_borders=False) -> None:
    super().__init__(csv_path, None, is_valid=is_valid)
    self.img_path_dir = Path(img_path_dir)
    assert self.img_path_dir.exists()
    
    self.remove_borders = remove_borders
  

  @staticmethod
  def _make_jng_synth(img_path_dict):
    return None
  

  def get_item_by_idx(self, idx):
    row = self.df.iloc[idx]
    path, annotation = itemgetter('Filename', 'Annotations')(row)
    img = cv2.imread(str(self.img_path_dir / path))
    
    if self.remove_borders:
      img = JeongganProcessor.remove_borders(img)
    
    return img, annotation


  def __getitem__(self, idx):
    img, annotation = self.get_item_by_idx(idx)
    img = self.transform(img)
    
    return img, self.tokenizer(annotation)
  

  def _make_tokenizer(self):
    return Tokenizer(self.df['Annotations'].values.tolist())
