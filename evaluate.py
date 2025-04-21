from time import time
from random import randint, choice, uniform, seed
from operator import itemgetter
from collections import defaultdict, Counter
import re

from pathlib import Path
import glob
import csv
import json
import pickle

from tqdm import tqdm
import matplotlib.pyplot as plt

from omegaconf import OmegaConf, DictConfig

import cv2
import numpy as np
import Levenshtein

import torch
from torch.utils.data import DataLoader

from exp_utils.model_zoo import TransformerOMR
from exp_utils.train_utils import Trainer, Tokenizer, pad_collate, LabelStudioDataset

filter_specials = lambda seq: [ x for x in seq if x not in (0, 1, 2) ]
tok_list_decode = lambda tknz, tl: [ tknz.vocab[t] for t in tl ]

def calc_acc(gt, prd):
  gt, prd = torch.tensor(gt, dtype=torch.long), torch.tensor(prd, dtype=torch.long)
  
  if prd.shape[0] > gt.shape[0]:
    prd = prd[:gt.shape[0]]
  elif prd.shape[0] < gt.shape[0]:
    prd = torch.cat([prd, torch.zeros(gt.shape[0] - prd.shape[0])], dim=-1)
  
  num_tokens = gt.shape[0]
  num_match_token = prd == gt
  num_match_token = num_match_token[gt != 0].sum().item()
  
  return num_match_token / num_tokens

def make_gt_pred_tuple_list(_test_set, _tokenizer, ls, with_img=False):
  tup_list = []
  
  for didx, pred in ls:
    img, label = _test_set.get_item_by_idx(didx)
    
    if not with_img:
      img = None
    
    label_enc = _tokenizer(label)[1:-1]
    pred = filter_specials(pred)
    
    acc = calc_acc(label_enc, pred)
    
    tup_list.append( (didx, img, label_enc, pred, acc) )
  
  return tup_list

def split_enc_seq(_tokenizer, enc_lb):
  blank_tok_idx = _tokenizer.tok2idx[' ']
  split_idx_list = [ i for i, t in enumerate(enc_lb) if t == blank_tok_idx ]
  
  is_ornament = lambda ti: '_' == _tokenizer.vocab[ti][0]
  is_position = lambda ti: ':' == _tokenizer.vocab[ti][0]
  
  def clean_findings(gl):
    il = [ i for i, t in enumerate(gl) if is_position(t) ]
    
    ls = []
    st = 0
    
    for i in il + [None]:
      ed = i + 1 if i else None
      
      sl = gl[st:ed]
      
      if len(sl) < 1:
        continue
      
      if is_ornament(sl[0]):
        sl = [0] + sl

      if not is_position(sl[-1]):
        sl = sl + [0]
      
      ls.append(sl)  
      
      st = ed
    
    return ls
  
  groups = []
  split_st = 0
  
  for si in split_idx_list + [None]:
    g = enc_lb[split_st:si]  
    g = clean_findings(g)
    
    for subg in g:
      note = subg[0]
      position = subg[-1]
      ornament = subg[1:-1]
    
      groups.append((note, position, ornament))
    
    if si:
      split_st = si + 1
  
  return groups

def make_category_lists(gl, num_ctg=6): # gl: [(note, pos, [orn])]
  n_l, p_l, np_l, o_l, op_l, onp_l = [ [] for _ in range(num_ctg) ]
  
  for g in gl:
    n, p, oo = g
    
    n_l.append(n)
    p_l.append(p)
    np_l.append((n, p))
    
    o_l += oo
    
    for o in oo:
      op_l.append((o, p))
      onp_l.append((o, n, p))
  
  n_l = filter_specials(n_l)
  p_l = filter_specials(p_l)
  
  return n_l, p_l, np_l, o_l, op_l, onp_l

def prepare_pairs(gt_gl, prd_gl, num_ctg=6): # x_gl: list of tuples
  
  gt_ctg_lists = make_category_lists(gt_gl, num_ctg=num_ctg)
  prd_ctg_lists = make_category_lists(prd_gl, num_ctg=num_ctg)
  
  return list(zip( gt_ctg_lists, prd_ctg_lists ))

def calc_metric(gt, prd): # +) [꾸밈](GT 꾸밈 0개 이상일 때), [꾸밈, 위치], [본음, 꾸밈, 위치]
  gt_tok_dict = defaultdict(int)
  
  for tok in gt:
    gt_tok_dict[tok] += 1
  
  num_match = 0
  
  for tok in prd:
    if gt_tok_dict[tok] > 0:
      num_match += 1
      gt_tok_dict[tok] -= 1
  
  eps = 1e-10
  
  precision = num_match / len(prd) if len(prd) > 0 else None
  recall = num_match / len(gt) if len(gt) > 0 else None
  
  f1 = 2 * (precision * recall) / (precision + recall + eps) if precision != None and recall != None else None
  
  return precision, recall, f1

def calc_all_metrics(_tokenizer, gt, prd):
  # gt_dec = _tokenizer.decode( gt )
  # prd_dec = _tokenizer.decode( prd )
  
  gt_split = split_enc_seq(_tokenizer, gt)
  prd_split = split_enc_seq(_tokenizer, prd)
  
  # (note_p, pos_p, note_pos_p, orn_p, orn_pos_p, orn_note_pos_p)
  # ((gt_note, prd_note), (gt_pos, prd_pos), (gt_note_pos, prd_note_pos), (gt_orn, prd_orn), (gt_orn_pos, prd_orn_pos), (gt_orn_note_pos, prd_orn_note_pos))
  pairs = prepare_pairs(gt_split, prd_split)
  
  # (ner, per, nper, oer, oper, onper)
  metric_values = [ calc_metric(*p) for p in pairs]
  
  return pairs, metric_values

def test(project_root_dir, exp_dir, csv_path):
  conf = OmegaConf.load(exp_dir / '.hydra' / 'config.yaml')
  device = torch.device(conf.general.device)
  
  model_dir = exp_dir / 'model'
  
  print('\n...tokenizer loading...')
  
  tokenizer_vocab_fn = model_dir / f'{conf.general.model_name}_tokenizer.txt'
  tokenizer = Tokenizer(vocab_txt_fn=tokenizer_vocab_fn)
  
  print('COMPLETE: load tokenizer')
  
  
  print('\n...data_set loading...')
  
  test_set = LabelStudioDataset(project_root_dir / conf.data_path.test, project_root_dir / 'jeongganbo-png/splited-pngs', remove_borders=conf.test_setting.remove_borders, is_valid=True)
  
  test_set.tokenizer = tokenizer
  
  test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, collate_fn=pad_collate, num_workers=conf.dataloader.num_workers_load)
  
  print('COMPLETE: data_set loading')
  
  
  print('\n...model initializing...')
  model = TransformerOMR(conf.model.dim, len(tokenizer.vocab), enc_depth=conf.model.enc_depth, dec_depth=conf.model.dec_depth, num_heads=conf.model.num_heads, dropout=conf.model.dropout)
  model.load_state_dict(torch.load(model_dir / f'{conf.general.model_name}_HL_{conf.test_setting.target_metric}_best.pt', map_location='cpu')['model'])

  tester = Trainer(
    model, 
    None, #optimizer
    None, #loss_fn
    None, #train_loader
    test_loader, 
    tokenizer,
    device=device, 
    scheduler=None,
    aux_loader=None,
    aux_freq=None,
    mix_aux=None,
    aux_valid_loader=None,
    wandb=None, 
    model_name=conf.general.model_name,
    model_save_path=model_dir,
    checkpoint_logger=None
  )

  print('COMPLETE: model initializing')
  

  print('\n...testing...')
  
  _, metric_dict, pred_tensor_list, _ = tester.validate(with_confidence=True)
  
  print('COMPLETE: Testing')
  
  
  print('\n...processing test result...')
  
  pred_list = []

  for b_idx in range(len(pred_tensor_list)):
    pred_list += pred_tensor_list[b_idx].tolist()

  pred_list = list( enumerate(pred_list) )
  pred_list = make_gt_pred_tuple_list(test_set, tokenizer, pred_list)
  
  print('COMPLETE: processing test result')
  
  print('\n...calculating metrics...')
  
  metric_names = ('ner', 'per', 'nper', 'oer', 'oper', 'opner', 'EMR', 'NMR', 'PMR', 'OrMR', 'dist', 'dist/len', 'insert', 'del', 'sub')
  num_metrics = len(metric_names)
  metric_ls = [ [] for _ in range(num_metrics) ]
  
  for case in pred_list:
    label, pred = case[2:4]
    
    # (note_p, pos_p, note_pos_p, orn_p, orn_pos_p, orn_note_pos_p)
    (note_pair, pos_pair, _, orn_pair, *_), metric_values = calc_all_metrics(tokenizer, label, pred)
    
    for m_idx, mv in enumerate(metric_values):
      metric_ls[m_idx].append(mv)
    
    is_note_match = int(Levenshtein.distance(note_pair[0], note_pair[1]) == 0)
    is_pos_match = int(Levenshtein.distance(pos_pair[0], pos_pair[1]) == 0)
    is_orn_match = int(Levenshtein.distance(orn_pair[0], orn_pair[1]) == 0)
    
    
    metric_ls[-9].append( (None, None, int(Levenshtein.distance(label, pred) == 0)) )
    metric_ls[-8].append( (None, None, is_note_match) )
    metric_ls[-7].append( (None, None, is_pos_match) )
    metric_ls[-6].append( (None, None, is_orn_match) )
    
    
    edit_ops = Levenshtein.editops(pred, label) # src, target order
    edit_dist = len(edit_ops)
    dist_len_ratio = edit_dist / len(label)
    edit_ops_cnt = Counter( [ op[0] for op in edit_ops ] )
    insert, delete, replace = itemgetter('insert', 'delete', 'replace')(edit_ops_cnt)
    
    for i, val in enumerate( (edit_dist, dist_len_ratio, insert, delete, replace) ):
      metric_ls[-5 + i].append( (None, None, val) )
  
  # print(len(pred_list))
  # print(len(metric_ls)) # should be 6
  # print(len(metric_ls[1])) # should be the same as num_total_case
  # print(len(metric_ls[0][0])) # 3
  # print()
  
  avg_metric_ls = []
  
  for mv_ls in metric_ls:
    avg_mv = []
    
    for subm_i in range(3):
      subm_ls = [ subm_t[subm_i] for subm_t in mv_ls if subm_t[subm_i] != None ]
      
      avg_subm = None
      
      if len(subm_ls) > 0:
        avg_subm = sum(subm_ls) / len(subm_ls)
      
      avg_mv.append( avg_subm )
    
    avg_metric_ls.append(tuple(avg_mv))

  # print(len(pred_list))
  # print(len(avg_metric_ls)) # should be 6
  # print(len(avg_metric_ls[1])) # 3
  # print()
  
  for m_name, avg_mv in zip(metric_names, avg_metric_ls):
    print(m_name, avg_mv)
  
  exact_all = metric_dict[conf.test_setting.target_metric]
  
  with open(csv_path, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow( [conf.general.model_name, exact_all] + [mv_t[-1] for mv_t in avg_metric_ls] )
  
  print('COMPLETE: calculating metrics')


if __name__ == "__main__":
  project_root_dir = Path('.')
  output_dir = project_root_dir / 'outputs'
  dir_date = '2024-05-04'
  dir_time = '20-31-38'

  exp_dir = output_dir / dir_date / dir_time  

  csv_path = project_root_dir / 'test' / 'new_metric_test.csv'

  test(project_root_dir, exp_dir, csv_path)