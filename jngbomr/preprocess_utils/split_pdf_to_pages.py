from pathlib import Path
from tqdm import tqdm
import pdfplumber

from ..jeonggan_utils import PIECE


def split_pdf_to_pages(jeongganbo_dir:Path):
  # List of instrument names
  instruments = PIECE.keys()

  scorebook_dir = jeongganbo_dir / 'scorebooks'
  split_page_dir = jeongganbo_dir / 'split_pages'
  split_page_dir.mkdir(exist_ok=True, parents=True)

  for instrument in instruments:
    pdf_path = scorebook_dir / f'{instrument}.pdf'
    
    if not pdf_path.exists():
      print(f"PDF file for {instrument} does not exist.")
      continue
    
    pdf = pdfplumber.open(str(pdf_path))

    for page in tqdm(pdf.pages):
      page_number = str(page.page_number).zfill(3)
      image = page.to_image(resolution=300)
      
      image_path = split_page_dir / f'{instrument}_pg-{page_number}.png'
      image.save(str(image_path))