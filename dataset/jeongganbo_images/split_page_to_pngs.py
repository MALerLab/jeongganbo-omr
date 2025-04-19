from pathlib import Path
from collections import defaultdict

import cv2

from jngbomr import JeongganboReader
from jngbomr import PIECE, START_PAGE


def main():
  preprocess_dir = Path(__file__).parent
  
  split_page_dir = preprocess_dir / 'split_pages'
  
  split_jeonggan_dir = preprocess_dir / 'split_jeonggans'
  split_jeonggan_dir.mkdir(exist_ok=True, parents=True)
  
  reader = JeongganboReader()
  pieces_by_instrument = {}
  instruments = PIECE.keys() 

  # Parse all pieces for each instrument
  for instrument in instruments:
    print(f"Parsing {instrument}...")
    page_paths = list(sorted(split_page_dir.glob(f'{instrument}_pg-*.png')))
    
    if len(page_paths) < 1:
      print(f"No pages found for {instrument}.")
      continue
    
    page_paths = page_paths[START_PAGE[instrument]:]
    pieces = reader.parse_multiple_pages(page_paths)
    pieces_by_instrument[instrument] = pieces

  piece_names = [
    piece_name 
    for instrument in instruments 
      for piece_name in PIECE[instrument]
  ]
  piece_names = list(set(piece_names))

  # Create a dictionary of pieces by piece name
  dict_by_piece_name = {}
  for piece_name in piece_names:
    dict_by_piece_name[piece_name] = defaultdict(dict)

  for inst in instruments:
    for title, piece in zip(PIECE[inst], pieces_by_instrument[inst]):
      dict_by_piece_name[title][inst] = piece

  # Check if all instruments have the same number of jeonggans for each piece
  clean_piece_names = []
  for piece in dict_by_piece_name:
    len_jeonggans = [
      len(dict_by_piece_name[piece][inst].jeonggans) 
      for inst in dict_by_piece_name[piece]
    ]
    
    if len(set(len_jeonggans)) == 1:
      clean_piece_names.append(piece)

  dict_by_clean_piece_name = {
    piece_name: dict_by_piece_name[piece_name] 
    for piece_name in clean_piece_names
  }

  # Save each jeonggan as a separate PNG file
  for piece_name, piece in dict_by_clean_piece_name.item():
    for inst in piece:
      parsed_piece = piece[inst]
      for jeonggan in parsed_piece.jeonggans:
        img_path = split_jeonggan_dir / f"{piece_name.replace(' ', '-')}_{inst}_{jeonggan.piece_beat}.png"
        cv2.imwrite(str(img_path), jeonggan.img)


if __name__ == "__main__":
  main()