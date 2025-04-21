"""
Example script to run the OMR model on individual jeonggan images.
"""

from pathlib import Path
import csv

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from jngbomr import Inferencer, INFERENCER_DEFAULT_KWARGS


def get_jng_img_sort_key(x):
  x = x.stem
  title, inst, idx = x.split('_')
  idx = int(idx)
  
  return (title, inst, idx)


def main(inferencer_config, out_path):
  inferencer = Inferencer(**inferencer_config)

  jng_img_paths = Path('dataset/jeongganbo/split_jeonggans').glob('*.png')
  jng_img_paths = sorted( jng_img_paths, key=get_jng_img_sort_key )

  data = jng_img_paths

  print(f"# of jeonggans: {len(data)}")

  batch_size = 100
  num_batch = round( len(data) / batch_size )

  if len(data) % batch_size > 0:
    num_batch += 1

  print(f"# of batches: {num_batch}")

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


if __name__ == '__main__': 
  main(
    inferencer_config=INFERENCER_DEFAULT_KWARGS,
    out_path='dataset/jeongganbo/omr_results_jeonggans.csv'
  )