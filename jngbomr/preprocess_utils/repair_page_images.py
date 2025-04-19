from pathlib import Path
from collections import defaultdict

import cv2

from ..jeonggan_utils import JeongganboReader
from ..jeonggan_utils import PAGE_REPAIRS


def repair_page_images(jeongganbo_images_dir:Path, overwrite:bool=False):
  split_pages_dir = jeongganbo_images_dir / 'split_pages'
  
  if not overwrite:
    save_dir = jeongganbo_images_dir /  'repaired_pages'
    save_dir.mkdir(parents=True, exist_ok=True)
  
  for page_fn, repairs  in PAGE_REPAIRS.items():
    page_path = split_pages_dir / f'{page_fn}.png'
    
    if not page_path.exists():
      print(f"Page {page_path} does not exist.")
      continue
    
    print(f"Processing {page_path}...")
    img = cv2.imread(str(page_path), cv2.IMREAD_UNCHANGED)
    
    for repair in repairs:
      repair_type = repair['type']
      vertices = repair['vertices']
      line_width = repair.get('line_width', 4)
      
      if repair_type == 'rect':
        img = cv2.rectangle(img, *vertices, (255, 255, 255), -1)
      
      elif repair_type == 'line':
        img = cv2.line(img, *vertices, (0, 0, 0), line_width)
    
    if not overwrite:
      save_path = save_dir / f'{page_fn}.png'
      cv2.imwrite(str(save_path), img)
      continue
    
    cv2.imwrite(str(page_path), img)