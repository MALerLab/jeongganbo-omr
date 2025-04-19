import re
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
  def __init__(
    self, 
    model, 
    optimizer, 
    loss_fn, 
    train_loader, 
    valid_loader, 
    tokenizer, 
    device,
    scheduler=None, 
    aux_loader=None,
    aux_freq=50,
    mix_aux=False,
    aux_valid_loader=None,
    wandb=None, 
    model_name='nmt_model', 
    model_save_path='model',
    checkpoint_logger=None
  ):

    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.loss_fn = loss_fn
    
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.aux_loader = aux_loader
    
    self.aux_freq = aux_freq
    self.mix_aux = mix_aux
    
    self.aux_valid_loader = aux_valid_loader
    self.tokenizer = tokenizer
    self.wandb = wandb
    self.checkpoint_logger = checkpoint_logger
    self.log_format = '%(metric_name)s, %(iter)s, %(acc)s'
    
    self.device = device
    self.model.to(device)
    
    self.training_loss = []
    
    self.grad_clip = 1.0
    
    self.valid_metrics = {
      'valid_acc': (0, []),
      'exact_all': (0, []),
      'exact_pos': (0, []),
      'exact_note': (0, []),
      'exact_length': (0, []),
    }
    
    self.HL_valid_metrics = {
      'valid_acc': (0, []),
      'exact_all': (0, []),
      'exact_pos': (0, []), 
      'exact_note': (0, []), 
      'exact_length': (0, []), 
    }
    
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
      # combining training sets
      if self.aux_loader and self.mix_aux:
        new_batch = []
        
        try :
          aux_batch = next(aux_iterator)
        except StopIteration:
          aux_iterator = iter(self.aux_loader)
          aux_batch = next(aux_iterator)
        
        for part_idx in range(len(batch)):
          train_partial_batch = batch[part_idx]
          aux_partial_batch = aux_batch[part_idx]
          
          # mix pad collate
          if part_idx > 0:
            max_token_len = max(train_partial_batch.shape[1], aux_partial_batch.shape[1])
            train_padded = torch.zeros((train_partial_batch.shape[0], max_token_len), dtype=torch.long)
            train_padded[:, :train_partial_batch.shape[1]] = train_partial_batch[:, :]
            train_partial_batch = train_padded
            
            aux_padded = torch.zeros((aux_partial_batch.shape[0], max_token_len), dtype=torch.long)
            aux_padded[:, :aux_partial_batch.shape[1]] = aux_partial_batch[:, :]
            aux_partial_batch = aux_padded
            
          cat_partial_batch = torch.cat((train_partial_batch, aux_partial_batch), dim=0)
          
          new_batch.append(cat_partial_batch)
        
        batch = tuple(new_batch)
      
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
      
      # aux train
      if self.aux_loader and not self.mix_aux and b_idx % self.aux_freq == 0:
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
      
      if (b_idx + 1) % 10_000 == 0:
        print(f'Saving checkpoint after {b_idx + 1} iterations')
        self.save_model(f'{self.model_name}_{b_idx + 1}.pt')
      
      # validation
      if (b_idx+1) % 500 == 0:
        # synthed
        self.model.eval()
        validation_acc, metric_dict = self.validate()
        metric_dict['valid_acc'] = validation_acc
        
        for metric_name in metric_dict.keys():
          if metric_name in ('note_pitch', 'note_pcalss'):
            continue
          
          metric_best, metric_list = self.valid_metrics[metric_name]
          metric_cur = metric_dict[metric_name]
          
          metric_list.append(metric_cur)
          
          if metric_cur > metric_best:
            print(f"best {metric_name} at iter #{ b_idx+1 }, Acc: {metric_cur:.4f}")
            self.checkpoint_logger.info(self.log_format % { 'metric_name': metric_name, 'iter': b_idx + 1, 'acc': metric_cur })  
            self.save_model(f'{self.model_name}_{metric_name}_best.pt')
          
          else:
            self.save_model(f'{self.model_name}_{metric_name}_last.pt')
            
          new_metric_best = max(metric_cur, metric_best)
          self.valid_metrics[metric_name] = (new_metric_best, metric_list)
        
        if self.wandb:
          self.wandb.log(metric_dict, step=b_idx)
        
        # HL
        self.model.eval()
        HL_validation_acc, HL_metric_dict = self.validate(external_loader=self.aux_valid_loader)
        HL_metric_dict['valid_acc'] = HL_validation_acc
        
        for metric_name in HL_metric_dict.keys():
          if metric_name in ('note_pitch', 'note_pcalss'):
            continue
          
          metric_best, metric_list = self.HL_valid_metrics[metric_name]
          metric_cur = HL_metric_dict[metric_name]
          
          metric_list.append(metric_cur)
          
          if metric_cur > metric_best:
            print(f"best HL {metric_name} at iter #{ b_idx+1 }, Acc: {metric_cur:.4f}")
            self.checkpoint_logger.info(self.log_format % { 'metric_name': f'HL_{metric_name}', 'iter': b_idx + 1, 'acc': metric_cur })  
            self.save_model(f'{self.model_name}_HL_{metric_name}_best.pt')
          
          else:
            self.save_model(f'{self.model_name}_HL_{metric_name}_last.pt')
            
          new_metric_best = max(metric_cur, metric_best)
          self.HL_valid_metrics[metric_name] = (new_metric_best, metric_list)
        
        HL_metric_dict = { f'HL_{key}': value for key, value in HL_metric_dict.items() }
        
        if self.wandb:
          self.wandb.log(HL_metric_dict, step=b_idx)
  

  def _train_by_single_batch(self, batch):
    src, tgt_i, tgt_o = batch
    pred = self.model(src.to(self.device), tgt_i.to(self.device))
    loss = self.loss_fn(pred, tgt_o)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    self.optimizer.step()
    if self.scheduler: self.scheduler.step()
    self.optimizer.zero_grad()
    
    return loss.item()


  # ONLY WORKS WITH NON-SHUFFLED SITUATION
  def validate(self, external_loader=None, with_confidence=False):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader

    self.model.eval()
    

    num_total_match_token = 0
    num_total_tokens = 0
    metric_dict_list = []
    
    if with_confidence:
      confidence_list = []
      pred_list = []
    
    
    with torch.no_grad():
      for batch in tqdm(loader, leave=False):
        src, tgt_i, tgt_o = batch
        
        pred = self.model.inference(src.to(self.device), max_len=tgt_o.shape[1], return_confidence=with_confidence)
        
        confidence = None
        
        if with_confidence:
          pred, confidence = pred
          confidence_list.append(confidence)
        
        pred = self.process_validation_outs(pred)
        if with_confidence:
          pred_list.append(pred)
        
        # match pred length to tgt_0 length
        if pred.shape[1] > tgt_o.shape[1]:
          pred = pred[:, :tgt_o.shape[1]]
        elif pred.shape[1] < tgt_o.shape[1]:
          pred = torch.cat([pred, torch.zeros((pred.shape[0], tgt_o.shape[1] - pred.shape[1]))], dim=1)
        
        num_tokens = (tgt_o != 0).sum().item()
        
        num_total_tokens += num_tokens
        num_match_token = pred.to(self.device) == tgt_o.to(self.device)
        num_match_token = num_match_token[tgt_o != 0].sum()
        num_total_match_token += num_match_token.item()
        
        metric_dict_list.append(self.calc_validation_acc(pred, tgt_o))
    
    validation_acc = num_total_match_token / num_total_tokens
    metric_dict = { 
      key: sum([ dct[key] for dct in metric_dict_list ]) / len(metric_dict_list)
      for key in metric_dict_list[0].keys() 
    }
    
    if with_confidence:
      return validation_acc, metric_dict, pred_list, confidence_list
    
    return validation_acc, metric_dict


  def process_validation_outs(self, pred):
    new_pred = []
    
    for prd in pred.clone().detach().cpu().numpy():
      end_indices, *_ = np.where(prd == 2)
      
      if end_indices.shape[0] > 1:
        end_index = end_indices[0] + 1
        prd[end_index:] = 0
      
      new_pred.append(prd[1:])
    
    new_pred = np.stack(new_pred, axis=0)
    new_pred = torch.tensor(new_pred, dtype=torch.long)
    
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