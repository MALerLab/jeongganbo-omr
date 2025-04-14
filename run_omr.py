from pathlib import Path
import csv

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from exp_utils.inferencer import Inferencer

def sort_key_jng_img(x):
  x = x.stem
  title, inst, idx = x.split('_')
  idx = int(idx)
  
  return (title, inst, idx)


exp_dir = '2024-05-04/11-10-10'
tokenizer_fn = 'transformer_4M_2_full_aux_tokenizer.txt'
weight_fn = 'transformer_4M_2_full_aux_HL_exact_all_best.pt'

exp_base = Path.home() / Path('userdata/jeongganbo-omr/outputs')

exp_path = exp_base / exp_dir
model_path = exp_path / 'model'
config_path = exp_path / '.hydra' / 'config.yaml'

out_path = model_path / ( weight_fn.split('.')[0] + '.csv' )

inferencer = Inferencer(
  vocab_txt_fn=(model_path / tokenizer_fn), 
  model_config_path=config_path,
  model_weights=(model_path / weight_fn),
  device='cuda:0'
)

jng_img_paths = Path('jeongganbo-png/splited-pngs/').glob('*.png')
jng_img_paths = [ i_p for i_p in jng_img_paths if 'yml_' not in str(i_p) ]
jng_img_paths = sorted( jng_img_paths, key=sort_key_jng_img )

data = jng_img_paths

print(len(data))

batch_size = 100
num_batch = round( len(data) / batch_size )

if len(data) % batch_size > 0:
  num_batch += 1

print(num_batch)

total = []

for i in tqdm( range(num_batch) ):
  batch_path = data[i * batch_size : (i+1) * batch_size]
  batch_img = [ cv2.imread(i_p) for i_p in batch_path ]
  
  pred_ls, _ = inferencer(batch_img)
  
  batch_path = [ i_p.name for i_p in batch_path ]
  total += list(zip( batch_path, pred_ls ))


with open(out_path, 'w', newline='', encoding='utf-8') as f:
  writer = csv.writer(f)
  writer.writerow([ 'filename', 'annotation' ])
  writer.writerows(total)