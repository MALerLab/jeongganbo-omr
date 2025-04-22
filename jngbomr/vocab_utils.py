import logging
import random
import re
from pathlib import Path
from operator import itemgetter
import math

from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

from .jeonggan_utils import JeongganProcessor, JeongganSynthesizer


class Tokenizer:
  def __init__(self, entire_strs=None, vocab_txt_fn=None) -> None:
    if vocab_txt_fn:
      with open(vocab_txt_fn, 'r') as f:
        self.vocab = [line for line in f.read().split('\n')]
      self.tok2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
    else:
      self.entire_strs = entire_strs
      self.vocab = self.get_vocab()
      self.tok2idx = {tok: idx for idx, tok in enumerate(self.vocab)}

  
  def tokenize_music_notation(self, notation):
    """
    Adjust the regex pattern to match:
    - single spaces as separate tokens
    - hyphens and colon-number pairs
    - sequences of non-space, non-colon characters not starting with an underscore
    - individual underscores followed by non-space, non-colon characters
    """
    pattern = r'( +|[^_\s:]+|_+[^_\s:]+|:\d+|[-])'

    # Use re.findall to extract all matching tokens, including spaces
    tokens = re.findall(pattern, notation)
    return tokens


  def get_vocab(self):
    # vocabs = {'\n', ','}
    vocabs = set()
    for label in self.entire_strs:
      words = self.tokenize_music_notation(label)
      vocabs.update(words)
    
    list_vocabs = sorted(list(vocabs))

    return ['<pad>', '<start>', '<end>'] + list_vocabs


  def __call__(self, label):
    words = self.tokenize_music_notation(label)
    words = ['<start>'] + words + ['<end>']

    return [self.tok2idx[word] for word in words]
  

  def __add__(self, other):
    return Tokenizer(self.entire_strs + other.entire_strs)


  def decode(self, labels):
    if isinstance(labels, torch.Tensor):
      if labels.ndim == 2:
        labels = labels.squeeze(0)
      
      labels = labels.tolist()
    
    if 2 in labels:
      labels = labels[:labels.index(2)]
    
    return ''.join([self.vocab[idx] for idx in labels if idx not in (0, 1, 2)])