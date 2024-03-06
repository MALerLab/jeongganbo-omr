import sys, getopt
import random
import re
from pathlib import Path
from operator import itemgetter
import csv

from omegaconf import OmegaConf
from tqdm.auto import tqdm
import pandas as pd
import cv2
import numpy as np

import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from exp_utils import JeongganSynthesizer, PNAME_EN_LIST, SYMBOL_W_DUR_EN_LIST


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
    return ''.join([self.vocab[idx] for idx in labels])

class Dataset:
  def __init__(self, csv_path, img_path_dict, is_valid=False) -> None:
    self.df = pd.read_csv(csv_path)
    self.img_path_dict = img_path_dict
    self.jng_synth = JeongganSynthesizer(img_path_dict)
    self.is_valid = is_valid

    if self.is_valid:
      self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Grayscale(num_output_channels=1),
      transforms.Lambda(lambda x: 1-x),
      transforms.Resize((160, 140)),
      ])
    else:
      self.transform = transforms.Compose([
      RandomBoundaryDrop(4),
      transforms.ToTensor(),
      transforms.Grayscale(num_output_channels=1),
      transforms.Lambda(lambda x: 1-x),
      transforms.RandomResizedCrop((160, 140), scale=(0.85, 1,0)),
      ])
    self.tokenizer = self._make_tokenizer()

  def _make_tokenizer(self):
    return Tokenizer(self.df['label'].values.tolist())
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    annotations, width, height = itemgetter('label', 'width', 'height')(row)
    img = None
    
    img = self.jng_synth.generate_image_by_label(annotations, width, height)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = self.transform(img)
    return img, self.tokenizer(annotations)
    

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


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    # self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.GELU()

  def forward(self, x):
    x = self.conv(x)
    # x = self.bn(x)
    x = self.relu(x)
    return x


class ContextAttention(nn.Module):
  def __init__(self, size, num_head):
    super(ContextAttention, self).__init__()
    self.attention_net = nn.Linear(size, size)
    self.num_head = num_head

    if size % num_head != 0:
      raise ValueError("size must be dividable by num_head", size, num_head)
    self.head_size = int(size/num_head)
    self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
    nn.init.uniform_(self.context_vector, a=-1, b=1)

  def get_attention(self, x):
    attention = self.attention_net(x)
    attention_tanh = torch.tanh(attention)
    attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
    similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
    similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
    return similarity

  def forward(self, x):
    attention = self.attention_net(x)
    attention_tanh = torch.tanh(attention)
    if self.head_size != 1:
      attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
      similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
      similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
      similarity[x.sum(-1)==0] = -1e10 # mask out zero padded_ones
      softmax_weight = torch.softmax(similarity, dim=1)

      x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
      weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1,1,1, x_split.shape[-1])
      attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
    else:
      softmax_weight = torch.softmax(attention, dim=1)
      attention = softmax_weight * x

    sum_attention = torch.sum(attention, dim=1)
    return sum_attention


class QKVAttention(nn.Module):
  def __init__(self, hidden_size, num_head=4, dropout=0.1):
    super().__init__()
    
    self.kv = nn.Linear(hidden_size, hidden_size*2)
    self.q = nn.Linear(hidden_size, hidden_size)
    self.num_head = num_head
    self.mlp = nn.Sequential(
      nn.Linear(hidden_size, hidden_size * 4),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_size * 4, hidden_size),
    )    
    
  def forward(self, _q, _kv):
    kv = self.kv(_kv)
    k, v = torch.split(kv, kv.shape[-1]//2, dim=-1)
    q = self.q(_q)
    attention_score = torch.bmm(q, k.permute(0,2,1))
    attention_score = torch.softmax(attention_score, dim=-1)
    attention = torch.bmm(attention_score, v)
    attention = self.mlp(attention)
    
    return attention
  


class OMRModel(nn.Module):
  def __init__(self, hidden_size, vocab_size, num_gru_layers=2):
    super().__init__()

    self.layers = nn.Sequential(
      ConvBlock(1, hidden_size//4, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size//4, hidden_size//4, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size//4, hidden_size//2, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size//2, hidden_size//2, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size//2, hidden_size, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size, hidden_size, 3, 1, 1),
    )

    self.context_attention = ContextAttention(hidden_size, 4)
    self.cont2hidden = nn.Linear(hidden_size, hidden_size*num_gru_layers)
    self.cnn_gru = nn.GRU(hidden_size, hidden_size//2, 1, batch_first=True, dropout = 0.2, bidirectional=True)

    self.emb = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, num_gru_layers, batch_first=True, dropout = 0.2)

    self.attn_layer1 = QKVAttention(hidden_size)
    self.final_gru1 = nn.GRU(hidden_size * 2, hidden_size, 1, batch_first=True)

    self.attn_layer2 = QKVAttention(hidden_size)
    self.final_gru2 = nn.GRU(hidden_size * 2, hidden_size*2, 1, batch_first=True)
    

    self.proj = nn.Linear(hidden_size*2, vocab_size)

  def run_img_cnn(self, x):
    x = self.layers(x)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    x = x.contiguous()
    x, _ = self.cnn_gru(x)

    context_vector = self.context_attention(x)
    context_vector = self.cont2hidden(context_vector.relu())
    context_vector = context_vector.reshape(x.shape[0], -1, context_vector.shape[-1]//2).permute(1,0,2)
    context_vector = context_vector.contiguous()

    return x, context_vector

  def forward(self, x, y):
    x, context_vector = self.run_img_cnn(x)
    y = self.emb(y)
    gru_out, _ = self.gru(y, context_vector)
    attention = self.attn_layer1(gru_out, x)
    cat_out = torch.cat([gru_out, attention], dim=-1)
    cat_out, _ = self.final_gru1(cat_out)
    
    attention = self.attn_layer2(cat_out, x)
    cat_out = torch.cat([cat_out, attention], dim=-1)
    cat_out, _ = self.final_gru2(cat_out)

    logit = self.proj(cat_out)

    return logit

  @torch.inference_mode()  
  def inference(self, x, max_len=None, batch=True):
    # assert x.shape[0] == 1 # batch size must be 1

    x, last_hidden = self.run_img_cnn(x)
    
    y = torch.ones((x.shape[0], 1), dtype=torch.long).to(x.device)
    outputs = torch.ones_like(y) if batch else []
    
    final_gru_last_hidden1 = None
    final_gru_last_hidden2 = None
    
    max_len = max_len if max_len else 100
    
    for _ in range(max_len):
      y = self.emb(y)
      gru_out, last_hidden = self.gru(y, last_hidden)
      
      attention = self.attn_layer1(gru_out, x)
      cat_out = torch.cat([gru_out, attention], dim=-1)
      cat_out, final_gru_last_hidden1 = self.final_gru1(cat_out, final_gru_last_hidden1)        
      
      attention = self.attn_layer2(cat_out, x)
      cat_out = torch.cat([cat_out, attention], dim=-1)
      cat_out, final_gru_last_hidden2 = self.final_gru2(cat_out, final_gru_last_hidden2)

      logit = self.proj(cat_out)
      y = torch.argmax(logit, dim=-1)
      
      if batch:
        outputs = torch.cat([outputs, y], dim=1) 
      else:
        if y.item() == 2:
          break
        outputs.append(y.item())
    
    return outputs



class Inferencer:
  def __init__(self, vocab_txt_fn='tokenizer.txt', model_weights='omr_model_best.pt'):
    self.tokenizer = Tokenizer(vocab_txt_fn=vocab_txt_fn)
    self.model = OMRModel(80, vocab_size=len(self.tokenizer.vocab), num_gru_layers=2)
    self.model.load_state_dict(torch.load(model_weights, map_location='cpu')['model'])
    self.model.eval()
    self.transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Grayscale(num_output_channels=1),
      transforms.Lambda(lambda x: 1-x),
      transforms.Resize((160, 140)),
      ])
  
  def __call__(self, img):
    if isinstance(img, str) or isinstance(img, Path):
      img = cv2.imread(img)
    img = self.transform(img)
    img = img.unsqueeze(0)
    pred = self.model.inference(img, batch=False)
    pred = self.tokenizer.decode(pred)
    return pred
  

class Trainer:
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, tokenizer, device, wandb=None, model_name='nmt_model', model_save_path='model'):

    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader
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
    for b_idx, batch in enumerate(tqdm(self.train_loader, leave=False)):
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
      
      if (b_idx+1) % 100 == 0:  
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
        end_index = end_indices[1]
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

def get_img_paths(img_path_base, sub_dirs):
  if isinstance(img_path_base, str):
    img_path_base = Path(img_path_base)
  
  paths = [ (img_path_base/sd).glob('*.png') for sd in sub_dirs ]
  paths = [ 
    p 
    for sd_p in paths 
    for p in sd_p 
  ]
  
  raw_dict = {
    str(p).split('/')[-1].replace('.png', ''): str(p) \
    for p in paths
  }

  res_dict = {}

  for name, path in sorted(raw_dict.items(), key=lambda x: x[0]):
    name = re.sub(r'(_\d\d\d)|(_ot)', '', name)
    
    if res_dict.get(name, False):
      res_dict[name].append(path)
    else:
      res_dict[name] = [path]

  for name, paths in res_dict.items():
    if len(paths) < 2:
      res_dict[name] = paths[0]

  return res_dict

def getConfs(argv):
  args = argv[1:]
  
  try:
    opt_list, _ = getopt.getopt(args, 'f:n:p:e:')
    
  except getopt.GetoptError:
    print('somethings gone wrong')
    return None

  opt_dict = {}
  
  for opt in opt_list:
    name, value = opt 
    name = name.replace('-', '')
    
    if value:
      opt_dict[name] = value
    else:
      opt_dict[name] = name
    
  conf = OmegaConf.load(opt_dict['f'])
  
  if opt_dict.get('e'):
    num_epoch_arg = int(opt_dict['e'])
    conf.num_epoch = num_epoch_arg
    
  if opt_dict.get('n'):
    name_arg = opt_dict['n']
    conf.run_name = name_arg
    
  if opt_dict.get('p'):
    project_arg = opt_dict['p']
    conf.project_name = project_arg
  
  return conf

def main(argv):
  conf = getConfs(argv)
  
  wandb_run = wandb.init(
    project=conf.project_name,
    name=conf.model_name,
    notes=conf.wandb_notes
  )
  
  print('\nSTART: data_set loading\n')
  
  note_img_path_dict = get_img_paths('test/synth/src', ['notes', 'symbols'])
  
  train_set = Dataset(conf.train_set_path, note_img_path_dict)
  valid_set = Dataset(conf.valid_set_path, note_img_path_dict, is_valid=True)
  
  train_loader = DataLoader(train_set, batch_size=conf.train_batch_size, shuffle=True, collate_fn=pad_collate)
  valid_loader = DataLoader(valid_set, batch_size=1000, shuffle=False, collate_fn=pad_collate)
  
  print('\nCOMPLETE: data_set loading\n')

  tokenizer = train_set.tokenizer + valid_set.tokenizer
  train_set.tokenizer = tokenizer
  valid_set.tokenizer = tokenizer
  
  with open(f'model/{conf.model_name}_tokenizer.txt', 'w') as f:
    f.write('\n'.join(tokenizer.vocab))

  model = OMRModel(80, vocab_size=len(tokenizer.vocab), num_gru_layers=2)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  trainer = Trainer(model, optimizer, get_nll_loss, train_loader, valid_loader, tokenizer, 'cuda', wandb=wandb_run, model_name=conf.model_name, model_save_path='model')
  
  print('\nStart: Training\n')
  
  trainer.train_and_validate()

if __name__ == "__main__":
  main(sys.argv)
