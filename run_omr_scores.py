from pathlib import Path
from typing import List
from collections import defaultdict

from jngbomr import JeongganboReader, Piece
from jngbomr import PIECE, START_PAGE
from jngbomr import INFERENCER_DEFAULT_KWARGS


class MultipleParser:
  def __init__(
    self,
    png_dir='dataset/jeongganbo/split_pages/',
    out_dir='dataset/jeongganbo/omr_results_scores/',
    piece_name_by_inst:dict=PIECE,
    start_page:dict=START_PAGE
  ) -> None:
    
    self.reader = JeongganboReader(
      run_omr=True, 
      inferencer_kwargs=INFERENCER_DEFAULT_KWARGS
    )
    self.png_dir = Path(png_dir)
    
    assert self.png_dir.exists(), f"Path {self.png_dir} does not exist."
    
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    
    self.piece_name_by_inst = piece_name_by_inst
    self.start_page = start_page
    self.inst_names = ['daegeum', 'piri', 'haegeum', 'ajaeng', 'gayageum', 'geomungo']
    
    self.pieces_by_instrument = {}
  
  
  def parse_all(self):
    for instrument in self.piece_name_by_inst.keys():
      self.pieces_by_instrument[instrument] = self.parse_instrument(instrument)
    
    self.check_title_assign()
    self.dict_by_piece_name = self.make_dict_by_piece_name()
    self.clean_piece_dict = self.check_jeonggan_length()
    
    return self.clean_piece_dict
  
  
  def parse_instrument(self, instrument:str) -> List[Piece]:
    print(f"Parsing {instrument}...")
    page_paths = sorted(list(self.png_dir.glob(f'{instrument}_pg-*.png')))
    page_paths = page_paths[self.start_page[instrument]:]
    
    pieces = self.reader.parse_multiple_pages(page_paths)
    
    assert len(['\n'.join(['|'.join([jg.omr_text for jg in gak.jeonggans]) for gak in piece.gaks if not gak.is_jangdan]) for piece in pieces]) > 0, f"No piece detected for {instrument}"
    
    return pieces
  
  
  def check_title_assign(self):
    for instrument in self.piece_name_by_inst.keys():
      assert len(self.pieces_by_instrument[instrument]) == len(self.piece_name_by_inst[instrument]), f"Num Detected Piece: {len(self.pieces_by_instrument[instrument])}, Num Title Assigned: {len(self.piece_name_by_inst[instrument])}"
    
    print("All title assigned correctly.")
    return
  
  
  def make_dict_by_piece_name(self):
    piece_names = [
      piece_name 
      for instrument in self.piece_name_by_inst 
      for piece_name in self.piece_name_by_inst[instrument]
    ]
    piece_names = list(set(piece_names))
    
    dict_by_piece_name = {}
    for piece_name in piece_names:
      dict_by_piece_name[piece_name] = defaultdict(dict)
    
    for inst in self.piece_name_by_inst.keys():
      for title, piece in zip(self.piece_name_by_inst[inst], self.pieces_by_instrument[inst]):
        dict_by_piece_name[title][inst] = piece
    
    return dict_by_piece_name
  
  
  def check_jeonggan_length(self):
    clean_piece_dict = {}
    for piece in self.dict_by_piece_name:
      len_jeonggans = [
        len(self.dict_by_piece_name[piece][inst].jeonggans) 
        for inst in self.dict_by_piece_name[piece]
      ]
      inst_names = [inst for inst in self.dict_by_piece_name[piece]]
      
      if len(set(len_jeonggans)) == 1 and len(len_jeonggans) > 2:
        print(f"{piece}: {len_jeonggans} / {inst_names}")
        clean_piece_dict[piece] = self.dict_by_piece_name[piece]
    
    return clean_piece_dict
  
  
  def save_omr_text(self, piece_dict, out_dir=None):  
    if out_dir is None:
      out_dir = self.out_dir
    
    for piece_name, inst_dict in piece_dict.items():
      piece_text = ''
      included_inst = []
      try:
        ['\n'.join(['|'.join([jg.omr_text for jg in gak.jeonggans]) for gak in piece.gaks if not gak.is_jangdan]) for piece in inst_dict.values()]
      
      except Exception as e:
        print(f"Error occured at {piece_name}")
        print(e)
        continue
      
      for inst in self.inst_names:
        if inst not in inst_dict: 
          continue
        
        piece = inst_dict[inst]
        omr_texts = [
          '|'.join([
            jg.omr_text 
            for jg in gak.jeonggans
          ]) 
          for gak in piece.gaks 
            if not gak.is_jangdan
        ]
        omr_texts = '\n'.join(omr_texts)
        piece_text += omr_texts + '\n\n'
        
        included_inst.append(inst)
      
      with open(Path(out_dir)/f'{piece_name}_{"_".join(included_inst)}.txt', 'w') as f:
        f.write(piece_text)



def main():
  parser = MultipleParser()
  piece_dict = parser.parse_all()
  parser.save_omr_text(piece_dict)


if __name__ == '__main__':
  main()