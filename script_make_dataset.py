from random import randint, choice
from time import time
import re
import csv
from pathlib import Path
import glob

from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt

from exp_utils import JeongganProcessor, JeongganSynthesizer
from omr_cnn import get_img_paths


NUM_RECIPE = 3_000_000
fail_cnt = 0
img_path_base = 'test/synth/src'
img_path_sub_dirs = ['notes', 'symbols']

img_paths = get_img_paths(img_path_base, img_path_sub_dirs)
jng_synth = JeongganSynthesizer(img_paths)

with open(f'data/train/006_all_{int(time())}.csv', 'w', newline='', encoding='utf-8') as f:
  writer = csv.writer(f)
  writer.writerow(['label', 'width', 'height'])
    
  for _ in tqdm(range(NUM_RECIPE)):
    
    while True:
      try:
        label, width, height, _ = jng_synth.generate_single_data()
        writer.writerow([label, width, height])
        break
      
      except:
        fail_cnt += 1
        # print(fail_cnt)

