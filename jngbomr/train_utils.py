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


def get_nll_loss(predicted_prob_distribution, indices_of_correct_token, eps=1e-10, ignore_index=0):
  if predicted_prob_distribution.ndim == 3:
    predicted_prob_distribution = predicted_prob_distribution.reshape(-1, predicted_prob_distribution.shape[-1])
    indices_of_correct_token = indices_of_correct_token.reshape(-1)

  prob_of_correct_next_word = torch.log_softmax(predicted_prob_distribution, dim=-1)[torch.arange(len(predicted_prob_distribution)), indices_of_correct_token]
  filtered_prob = prob_of_correct_next_word[indices_of_correct_token != ignore_index]
  loss = -filtered_prob
  return loss.mean()



class CosineLRScheduler(_LRScheduler):
  """
  Cosine LR scheduler.
  Args:
    optimizer (Optimizer): Torch optimizer.
    warmup_steps (int): Number of warmup steps.
    total_steps (int): Total number of steps.
    lr_min_ratio (float): Minimum learning rate.
    cycle_length (float): Cycle length.
  """
  def __init__(
    self, 
    optimizer:Optimizer, 
    total_steps:int, 
    warmup_steps:int,
    lr_min_ratio:float=0.0,
    cycle_length:float=1.0
  ):
    self.warmup_steps = warmup_steps
    assert self.warmup_steps >= 0
    self.total_steps = total_steps
    assert self.total_steps >= 0
    self.lr_min_ratio = lr_min_ratio
    self.cycle_length = cycle_length
    super().__init__(optimizer)


  def _get_sched_lr(self, lr: float, step: int):
    if step < self.warmup_steps:
      lr_ratio = step / self.warmup_steps
      lr = lr_ratio * lr
    elif step <= self.total_steps:
      s = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
      lr_ratio = self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * \
        (1. + math.cos(math.pi * s / self.cycle_length))
      lr = lr_ratio * lr
    else:
      lr_ratio = self.lr_min_ratio
      lr = lr_ratio * lr
    return lr


  def get_lr(self):
    return [self._get_sched_lr(lr, self.last_epoch) for lr in self.base_lrs]



# draw bottom 30 jngs based on confidence
def draw_low_confidence_plot(dataset, confidence_tensor_list, pred_tensor_list):
  confidence_list = []
  pred_list = []
  
  for b_idx in range(len(pred_tensor_list)):
    confidence_list += confidence_tensor_list[b_idx].tolist()
    pred_list += pred_tensor_list[b_idx].tolist()
  
  confidence_list = sorted(zip(confidence_list, [i for i in range(len(confidence_list))]), key=lambda x: x[0])
  
  plt.rcParams.update({'font.family': 'NanumGothic'})
  fig = plt.figure(figsize=(30, 40), layout='tight', dpi=150.0)
  
  subplot_idx = 1
  
  for conf, data_idx in confidence_list[:30]:
    img, annotation = dataset.get_item_by_idx(data_idx)
    
    prd = pred_list[data_idx]
    prd_dec = dataset.tokenizer.decode( list(map(int, filter(lambda x: x not in (0, 1, 2), prd))) )
    
    plt.subplot(5, 6, subplot_idx)
    plt.imshow(img)
    plt.title(f'{round(conf, 3)}\n{annotation}\n{prd_dec}', loc='left', fontsize=10)
    
    subplot_idx += 1
  
  fig.canvas.draw()
  
  plt.close()
  
  return np.array(fig.canvas.renderer._renderer)