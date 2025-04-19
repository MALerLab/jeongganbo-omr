import shutil
from tqdm.auto import tqdm
from pathlib import Path
import cv2
import easyocr

reader = easyocr.Reader(['ch_tra'], gpu=True)

# Read pngs
save_dir = Path('jeongganbo-png/splited-pngs/')
pngs = list(save_dir.glob('*.png'))
len(pngs)

unique_dir = Path('jeongganbo-png/unique-char-pngs/')
unique_dir.mkdir(exist_ok=True, parents=True)
chars = set()
unique_pngs = []

for png in tqdm(pngs):
  img = cv2.imread(str(png))
  result = reader.readtext(img)
  result_char = '+'.join([x[1] for x in result])
  if result_char not in chars:
    chars.add(result_char)
    unique_pngs.append(png)
    # Move the PNG file to unique_dir directory
    shutil.move(str(png), str(unique_dir / png.name))


