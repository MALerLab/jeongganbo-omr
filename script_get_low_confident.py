import argparse
from typing import List, Union
from pathlib import Path
from tqdm.auto import tqdm
import shutil

import cv2

from exp_utils.inferencer import Inferencer

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--jgb_dir', type=str, default='jeongganbo-png/splited-pngs/')
  parser.add_argument('--output_dir', type=str, default='jeongganbo-png/low_confident_pngs')
  parser.add_argument('--model_path', type=str, default='model/transformer_240324_best.pt')
  return parser
  
  
def main():
  args = get_parser().parse_args()
  
  reader = Inferencer(vocab_txt_fn='_'.join(args.model_path.split('_')[:-1])+'_tokenizer.txt',
                      model_weights=args.model_path, device='cuda')

  
  png_dir = Path(args.jgb_dir)
  pngs = list(png_dir.glob('*.png')) + list(Path('jeongganbo-png/unique-char-pngs').glob('*.png'))
  print(len(pngs))

  output_dir = Path(args.output_dir)
  output_dir.mkdir(exist_ok=True, parents=True)
  chars = set()
  
  batch_size = 1000

  for idx in tqdm(range(0, len(pngs), batch_size)):
    batch_pngs = pngs[idx:idx+batch_size]
    imgs = [cv2.imread(str(png)) for png in batch_pngs]
    result, confident = reader(imgs)
    for png, text, conf in zip(batch_pngs, result, confident):
      if conf < 0.3:
        if text not in chars:
          chars.add(text)
          # Copy the PNG file to unique_dir directory
          shutil.copy(str(png), str(output_dir / png.name))


if __name__ == "__main__":
  main()