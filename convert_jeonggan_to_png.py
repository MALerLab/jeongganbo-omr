from pathlib import Path
from collections import defaultdict
from data_utils import JeongganboReader
from constants import PIECE, START_PAGE




def main():
  png_dir = Path('jeongganbo-png/pngs/')
  reader = JeongganboReader()
  pieces_by_instrument = {}
  instruments = PIECE.keys() 

  # Parse all pieces for each instrument
  for instrument in instruments:
    print(f"Parsing {instrument}...")
    png_fns = sorted(list(png_dir.glob(f'{instrument}_pg-*.png')))
    pieces = reader.parse_multiple_pages(png_fns[START_PAGE[instrument]:])
    pieces_by_instrument[instrument] = pieces

  piece_names = list(set([piece_name for instrument in instruments for piece_name in PIECE[instrument]]))


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
    len_jeonggans = [len(dict_by_piece_name[piece][inst].jeonggans) for inst in dict_by_piece_name[piece]]
    inst_names = [inst for inst in dict_by_piece_name[piece]]
    if len(set(len_jeonggans)) == 1:
      # print(f"{piece}: {len_jeonggans} / {inst_names}")
      clean_piece_names.append(piece)

  dict_by_clean_piece_name = {piece_name: dict_by_piece_name[piece_name] for piece_name in clean_piece_names}


  return dict_by_clean_piece_name

if __name__ == "__main__":
  main()