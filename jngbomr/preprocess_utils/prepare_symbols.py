from pathlib import Path
from collections import defaultdict

import csv

import numpy as np
import cv2


def crop_image(img, threshold=100):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
  
  img_grey = 255 - cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
  
  slice_idxs = []
  
  for iter_num in range(4):
    target = np.transpose(img_grey, axes=(1, 0)) if iter_num > 1 else img_grey
      
    rng = range(len(target)) if iter_num % 2 == 0 else range(len(target) - 1, 0, -1)
    
    for idx in rng:
      arr = target[idx]
      p_sum = sum(arr)
      
      if p_sum > threshold:
        slice_idxs.append(idx)
        break
  
  row_st, row_ed, col_st, col_ed = slice_idxs
  
  return img[row_st:row_ed+1, col_st:col_ed+1, :]


def prepare_symbols(jeongganbo_dir:Path):
  with open(jeongganbo_dir / 'where_to_find_your_symbols.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    symbol_locations = list(reader)[1:]
  
  symbol_locations = [
    [ filename, jeongganbo_dir / src, eval(bbox) ]
    for filename, src, bbox in symbol_locations
  ]
  
  symbol_dir = jeongganbo_dir / 'synth'
  symbol_dir.mkdir(parents=True, exist_ok=True)
  
  (symbol_dir / 'notes').mkdir(parents=True, exist_ok=True)
  (symbol_dir / 'symbols').mkdir(parents=True, exist_ok=True)
  
  for filename, src, bbox in symbol_locations:
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    x1, y1, x2, y2 = bbox
    img = img[y1:y2, x1:x2]
    
    img = crop_image(img)
    
    save_path = symbol_dir / filename
    
    cv2.imwrite(str(save_path), img)