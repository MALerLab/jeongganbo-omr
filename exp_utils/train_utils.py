import random
import re
from pathlib import Path
from operator import itemgetter
import math

from tqdm.auto import tqdm
import pandas as pd
import cv2
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from exp_utils import JeongganSynthesizer, PNAME_EN_LIST, SYMBOL_W_DUR_EN_LIST, JeongganProcessor
from exp_utils.jeonggan_synthesizer import get_img_paths
from exp_utils.model_zoo import OMRModel, TransformerOMR

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class Trainer:
  def __init__(self, model, 
               optimizer, 
               loss_fn, 
               train_loader, 
               valid_loader, 
               tokenizer, 
               device,
               scheduler=None, 
               aux_loader=None,
               wandb=None, 
               model_name='nmt_model', 
               model_save_path='model'):

    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.aux_loader = aux_loader
    self.tokenizer = tokenizer
    self.wandb = wandb

    
    self.model.to(device)
    
    self.grad_clip = 1.0
    self.best_valid_accuracy = 0
    self.device = device
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []
    self.model_name = model_name 
    self.model_save_path = Path(model_save_path)

  def save_model(self, path):
    torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, self.model_save_path / path)
  
  def save_loss_acc(self, loss_acc_dict, path):
    torch.save(loss_acc_dict, self.model_save_path / path)
    
  def train_and_validate(self):
    if self.aux_loader:
      aux_iterator = iter(self.aux_loader)
    for b_idx, batch in enumerate(tqdm(self.train_loader)):
      self.model.train()
      loss_value = self._train_by_single_batch(batch)
      self.training_loss.append(loss_value)
      
      if self.wandb:
        self.wandb.log(
          {
            'loss_train': loss_value
          },
          step=b_idx
        )
        
      if self.aux_loader and b_idx % 50 == 0:
        try :
          aux_batch = next(aux_iterator)
        except StopIteration:
          aux_iterator = iter(self.aux_loader)
          aux_batch = next(aux_iterator)
        aux_loss = self._train_by_single_batch(aux_batch)
        if self.wandb:
          self.wandb.log(
            {
              'loss_aux': aux_loss
            },
            step=b_idx
          )
      if (b_idx+1) % 500 == 0:  
        self.model.eval()
        validation_loss, validation_acc, metric_dict = self.validate()
        #self.validation_loss.append(validation_loss)
        self.validation_acc.append(validation_acc)
    
        if validation_acc > self.best_valid_accuracy:
          print(f"Saving the model with best validation accuracy: valid #{(b_idx+1) // 100}, Acc: {validation_acc:.4f} ")
          self.save_model(f'{self.model_name}_best.pt')
        else:
          self.save_model(f'{self.model_name}_last.pt')
          
        self.best_valid_accuracy = max(validation_acc, self.best_valid_accuracy)
        
        # metric_dict['valid_loss'] = validation_loss
        metric_dict['valid_acc'] = validation_acc
        
        if self.wandb:
          self.wandb.log(metric_dict, step=b_idx)
  
    self.save_loss_acc({ 'train_loss': self.training_loss, 'valid_loss': self.validation_loss, 'valid_acc': self.validation_acc}, f'{self.model_name}_loss_acc.pt')
  
  def _train_by_single_batch(self, batch):
    '''
    This method updates self.model's parameter with a given batch
    
    batch (tuple): (batch_of_input_text, batch_of_label)
    
    You have to use variables below:
    
    self.model (Translator/torch.nn.Module): A neural network model
    self.optimizer (torch.optim.adam.Adam): Adam optimizer that optimizes model's parameter
    self.loss_fn (function): function for calculating BCE loss for a given prediction and target
    self.device (str): 'cuda' or 'cpu'

    output: loss (float): Mean binary cross entropy value for every sample in the training batch
    The model's parameters, optimizer's steps has to be updated inside this method
    '''
    
    src, tgt_i, tgt_o = batch
    pred = self.model(src.to(self.device), tgt_i.to(self.device))
    loss = self.loss_fn(pred, tgt_o)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    self.optimizer.step()
    if self.scheduler: self.scheduler.step()
    self.optimizer.zero_grad()
    
    return loss.item()

    
  def validate(self, external_loader=None):
    '''
    This method calculates accuracy and loss for given data loader.
    It can be used for validation step, or to get test set result
    
    input:
      data_loader: If there is no data_loader given, use self.valid_loader as default.
      
    output: 
      validation_loss (float): Mean Binary Cross Entropy value for every sample in validation set
      validation_accuracy (float): Mean Accuracy value for every sample in validation set
    '''
    
    ### Don't change this part
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader

    self.model.eval()
    
    '''
    Write your code from here, using loader, self.model, self.loss_fn.
    '''
    validation_loss = 0
    validation_acc = 0
    num_total_tokens = 0
    with torch.no_grad():
      for batch in tqdm(loader, leave=False):
        
        src, tgt_i, tgt_o = batch
        pred = self.model.inference(src.to(self.device), max_len=tgt_o.shape[1])
        
        pred = self.process_validation_outs(pred)
        if pred.shape[1] > tgt_o.shape[1]:
          pred = pred[:, :tgt_o.shape[1]]
        elif pred.shape[1] < tgt_o.shape[1]:
          pred = torch.cat([pred, torch.zeros((pred.shape[0], tgt_o.shape[1] - pred.shape[1]))], dim=1)
        
        # loss = self.loss_fn(pred, tgt_o)
        num_tokens = (tgt_o != 0).sum().item()
        # validation_loss += loss.item() * num_tokens
        
        num_total_tokens += num_tokens
        acc_exact = pred.to(self.device) == tgt_o.to(self.device)
        acc_exact = acc_exact[tgt_o != 0].sum()
        validation_acc += acc_exact.item()
        
        metric_dict = self.calc_validation_acc(pred, tgt_o)
        
    return validation_loss / num_total_tokens, validation_acc / num_total_tokens, metric_dict

  def process_validation_outs(self, pred):
    new_pred = []
    
    for prd in pred.clone().detach().cpu().numpy():
      end_indices, *_ = np.where(prd == 2)
      
      if end_indices.shape[0] > 1:
        end_index = end_indices[0] + 1 # Doesn't it have to be end_indices[0] + 1 instead of end_indices[1]?
        prd[end_index:] = 0
      
      new_pred.append(prd[1:])
    
    new_pred = np.stack(new_pred, axis=0)
    new_pred = torch.tensor(new_pred, dtype=torch.float64)
    
    return new_pred
  
  def calc_validation_acc(self, pred, tgt_o):
    num_total = pred.shape[0]
    cnts = [0, 0, 0, 0] # [exact, position, notes, length]

    length_match_token_cnt = 0
    add_cnts = [0, 0] # [exact, pclass]
    
    split_octave_and_pclass = lambda string: re.findall(r'(배|하배|하하배|청|중청)?(.+)', string)[0]
    
    pred = pred.clone().detach().cpu().tolist()
    tgt_o = tgt_o.clone().detach().cpu().tolist()
    
    for prd, tar in zip(pred, tgt_o):
      prd_filtered = self.tokenizer.decode( list(map(int, filter(lambda x: x not in (0, 1, 2), prd))) )
      tar_filtered = self.tokenizer.decode( list(map(int, filter(lambda x: x not in (0, 1, 2), tar))) )
      
      if tar_filtered == prd_filtered:
        cnts[0] += 1
      
      tar_notes, tar_positions = self.get_notes_and_positions(prd_filtered)
      prd_notes, prd_positions = self.get_notes_and_positions(tar_filtered)
      
      if len(tar_notes) == len(prd_notes):
        cnts[3] += 1
        length_match_token_cnt += len(tar_notes)
        
        for note, hnote in zip(tar_notes, prd_notes):
          note_oct, note_pc = split_octave_and_pclass(note)
          hnote_oct, hnote_pc = split_octave_and_pclass(hnote)
          
          if note_oct == hnote_oct and note_pc == hnote_pc:
            add_cnts[0] += 1
            
          if note_pc == hnote_pc:
            add_cnts[1] += 1
      
      if tar_positions == prd_positions:
        cnts[1] += 1
      
      if tar_notes == prd_notes:
        cnts[2] += 1
    
    cnts = [ cnt / num_total for cnt in cnts ]
    add_cnts = [ a_cnt / length_match_token_cnt if length_match_token_cnt > 0 else 0 for a_cnt in add_cnts ] 
    
    return { key: value for key, value in zip(['exact_all', 'exact_pos', 'exact_note', 'exact_length', 'note_pitch', 'note_pcalss'], cnts+add_cnts) }
  
  @staticmethod
  def get_notes_and_positions(label):
    pattern = r'([^_\s:]+|_+[^_\s:]+|[^:]\d+|[-])'
    
    token_groups = label.split()
    
    notes = []
    positions = []
    
    for group in token_groups:
      findings = re.findall(pattern, group)
      notes.append(findings[0])
      positions.append(findings[-1])
    
    return notes, positions

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
    # Adjust the regex pattern to match:
    # - single spaces as separate tokens
    # - hyphens and colon-number pairs
    # - sequences of non-space, non-colon characters not starting with an underscore
    # - individual underscores followed by non-space, non-colon characters
    pattern = r'( +|[^_\s:]+|_+[^_\s:]+|:\d+|[-])'

    # Use re.findall to extract all matching tokens, including spaces
    tokens = re.findall(pattern, notation)
    return tokens

  def get_vocab(self):
    # vocabs = {'\n', ','}
    vocabs = set()
    for label in self.entire_strs:
      # words = re.split(' |\n', label)
      # words = [word.replace(',', '') for word in words]
      # words = [word for word in words if word != '']
      # vocabs.update(words)
      words = self.tokenize_music_notation(label)
      vocabs.update(words)
    list_vocabs = sorted(list(vocabs))
    return ['<pad>', '<start>', '<end>'] + list_vocabs

  def __call__(self, label):
    # label = label.replace('\n', ' \n ')
    # label = label.replace(',', ' , ')
    # words = label.split(' ')
    # words = [word for word in words if word != '']
    words = self.tokenize_music_notation(label)
    words = ['<start>'] + words + ['<end>']
    return [self.tok2idx[word] for word in words]
  
  def __add__(self, other):
    return Tokenizer(self.entire_strs + other.entire_strs)

  def decode(self, labels):
    if isinstance(labels, torch.Tensor):
      labels = labels.tolist()
    if 2 in labels:
      labels = labels[:labels.index(2)]
    return ''.join([self.vocab[idx] for idx in labels if idx not in (0, 1, 2)])

class Dataset:
  def __init__(self, csv_path, img_path_dict, is_valid=False) -> None:
    self.df = pd.read_csv(csv_path)
    self.img_path_dict = img_path_dict
    self.jng_synth = self._make_jng_synth(img_path_dict)
    self.is_valid = is_valid
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
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    annotations, width, height = itemgetter('label', 'width', 'height')(row)
    img = None
    
    while True:
      try:
        img = self.jng_synth.generate_image_by_label(annotations, width, height, apply_noise=self.need_random, random_symbols=self.need_random, layout_elements=self.need_random)
        break
      except:
        pass
    
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = self.transform(img)
    return img, self.tokenizer(annotations)

class LabelStudioDataset(Dataset):
  def __init__(self, csv_path, img_path_dir, is_valid=False) -> None:
    super().__init__(csv_path, None, is_valid)
    self.img_path_dir = Path(img_path_dir)
    assert self.img_path_dir.exists()
    
    
  @staticmethod
  def _make_jng_synth(img_path_dict):
    return None

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    path, annotation = itemgetter('Filename', 'Annotations')(row)
    img = cv2.imread(str(self.img_path_dir / path))
    img = self.transform(img)
    return img, self.tokenizer(annotation)
  
  def _make_tokenizer(self):
    return Tokenizer(self.df['Annotations'].values.tolist())


def pad_collate(raw_batch):
  # raw batch is a list of tuples (img, annotation)
  # img is torch tensor with shape (1, H, W)
  # pad to same width by adding 0s to the left and right
  # pad to same height by adding 0s to the top and bottom

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


def get_nll_loss(predicted_prob_distribution, indices_of_correct_token, eps=1e-10, ignore_index=0):
  '''
  for PackedSequence, the input is 2D tensor
  
  predicted_prob_distribution has a shape of [num_entire_tokens_in_the_batch x vocab_size]
  indices_of_correct_token has a shape of [num_entire_tokens_in_the_batch]
  '''

  if predicted_prob_distribution.ndim == 3:
    predicted_prob_distribution = predicted_prob_distribution.reshape(-1, predicted_prob_distribution.shape[-1])
    indices_of_correct_token = indices_of_correct_token.reshape(-1)


  prob_of_correct_next_word = torch.log_softmax(predicted_prob_distribution, dim=-1)[torch.arange(len(predicted_prob_distribution)), indices_of_correct_token]
  filtered_prob = prob_of_correct_next_word[indices_of_correct_token != ignore_index]
  loss = -filtered_prob
  return loss.mean()


class CosineLRScheduler(_LRScheduler):
    """Cosine LR scheduler.
    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps.
        lr_min_ratio (float): Minimum learning rate.
        cycle_length (float): Cycle length.
    """
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: int,
                 lr_min_ratio: float = 0.0, cycle_length: float = 1.0):
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