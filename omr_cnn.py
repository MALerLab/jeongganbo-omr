import torch
import torch.nn as nn
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import pandas as pd
import cv2
import re
from pathlib import Path


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
  def __init__(self, entire_strs) -> None:
    self.entire_strs = entire_strs
    self.vocab = self.get_vocab()
    self.tok2idx = {tok: idx for idx, tok in enumerate(self.vocab)}

  def get_vocab(self):
    vocabs = {'\n', ','}
    for label in self.entire_strs:
      words = re.split(' |\n', label)
      words = [word.replace(',', '') for word in words]
      words = [word for word in words if word != '']
      vocabs.update(words)
    list_vocabs = sorted(list(vocabs))
    return ['<pad>', '<start>', '<end>'] + list_vocabs

  def __call__(self, label):
    label = label.replace('\n', ' \n ')
    label = label.replace(',', ' , ')
    words = label.split(' ')
    words = [word for word in words if word != '']
    words = ['<start>'] + words + ['<end>']
    return [self.tok2idx[word] for word in words]
  
  def decode(self, labels):
    return ' '.join([self.vocab[idx] for idx in labels])

class Dataset:
  def __init__(self, csv_path, img_dir) -> None:
    self.df = pd.read_csv(csv_path)
    self.img_dir = Path(img_dir)
    self.transform = transforms.Compose([
    RandomBoundaryDrop(4),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: 1-x),
    # transforms.Resize((160, 140)),
    ])
    self.tokenizer = Tokenizer(self.df['Annotations'].values.tolist())

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    img = cv2.imread( str(self.img_dir / row['Filename']))
    annotations = row['Annotations']
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

    self.kv = nn.Linear(hidden_size, hidden_size*2)
    self.q = nn.Linear(hidden_size, hidden_size)
    self.num_head = 4

    self.emb = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, num_gru_layers, batch_first=True, dropout = 0.2)
    self.mlp = nn.Sequential(
        nn.Linear(hidden_size, hidden_size * 4),
        nn.GELU(),
        nn.Linear(hidden_size * 4, hidden_size),
    )

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
    kv = self.kv(x)
    k, v = torch.split(kv, kv.shape[-1]//2, dim=-1)
    q = self.q(gru_out)
    attention_score = torch.bmm(q, k.permute(0,2,1))
    attention_score = torch.softmax(attention_score, dim=-1)

    attention = torch.bmm(attention_score, v)
    attention = self.mlp(attention)

    cat_out = torch.cat([gru_out, attention], dim=-1)
    logit = self.proj(cat_out)

    return logit

  @torch.inference_mode()  
  def inference(self, x):
    assert x.shape[0] == 1 # batch size must be 1

    x, last_hidden = self.run_img_cnn(x)
    kv = self.kv(x)
    k, v = torch.split(kv, kv.shape[-1]//2, dim=-1)

    y = torch.ones((1, 1), dtype=torch.long).to(x.device)
    outputs = []
    for _ in range(100):
      y = self.emb(y)
      gru_out, last_hidden = self.gru(y, last_hidden)
      q = self.q(gru_out)
      attention_score = torch.bmm(q, k.permute(0,2,1))
      attention_score = torch.softmax(attention_score, dim=-1)

      attention = torch.bmm(attention_score, v)
      attention = self.mlp(attention)

      cat_out = torch.cat([gru_out, attention], dim=-1)
      logit = self.proj(cat_out)
      y = torch.argmax(logit, dim=-1)
      if y == 2:
        break
      outputs.append(y.item())
    return outputs
class Trainer:
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device, model_name='nmt_model'):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    
    self.model.to(device)
    
    self.grad_clip = 1.0
    self.best_valid_accuracy = 0
    self.device = device
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []
    self.model_name = model_name

  def save_model(self, path):
    torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, path)
    
  def train_by_num_epoch(self, num_epochs):
    for epoch in tqdm(range(num_epochs)):
      self.model.train()
      for batch in tqdm(self.train_loader, leave=False):
        loss_value = self._train_by_single_batch(batch)
        self.training_loss.append(loss_value)
      self.model.eval()
      validation_loss, validation_acc = self.validate()
      self.validation_loss.append(validation_loss)
      self.validation_acc.append(validation_acc)
      
      if validation_acc > self.best_valid_accuracy:
        print(f"Saving the model with best validation accuracy: Epoch {epoch+1}, Acc: {validation_acc:.4f} ")
        self.save_model(f'{self.model_name}_best.pt')
      else:
        self.save_model(f'{self.model_name}_last.pt')
      self.best_valid_accuracy = max(validation_acc, self.best_valid_accuracy)

      
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
        pred = self.model(src.to(self.device), tgt_i.to(self.device))
        loss = self.loss_fn(pred, tgt_o)
        num_tokens = (tgt_o != 0).sum().item()
        validation_loss += loss.item() * num_tokens
        num_total_tokens += num_tokens
        
        acc = torch.argmax(pred, dim=-1) == tgt_o.to(self.device)
        acc = acc[tgt_o != 0].sum()
        validation_acc += acc.item()
        
    return validation_loss / num_total_tokens, validation_acc / num_total_tokens

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