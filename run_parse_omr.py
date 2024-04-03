import re
import pickle
import argparse
from typing import List, Union

from fractions import Fraction
from pathlib import Path
from collections import OrderedDict
from data_utils import JeongganboReader, Jeonggan, Piece
from music21 import stream, note as mnote, meter as mmeter, key as mkey, pitch as mpitch

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--jgb_dir', type=str, default='jeongganbo-png/pngs/')
  parser.add_argument('--output_dir', type=str, default='omr_results/')
  parser.add_argument('--model_path', type=str, default='model/transformer_240324_best.pt')
  parser.add_argument('--error_dir', type=str, default='low_conf_jg_dir/')
  return parser
  
  
def define_pitch2midi():
  pitch2midi = {}
  pitch_name = ["황" ,'대' ,'태' ,'협' ,'고' ,'중' ,'유' ,'임' ,'이' ,'남' ,'무' ,'응']
  octave_name = {"하하배": -3, "하배": -2, "배":-1, '':0, '청':+1, '중청':+2}

  for o in octave_name.keys():
    for i, p in enumerate(pitch_name):
      pitch2midi[o+p] = i + 63 + octave_name[o] * 12
  return pitch2midi

PITCH2MIDI = define_pitch2midi()


class Symbol:
  def __init__(self, text:str, offset=0):
    # example of text: '배임:5' or '-:8', '임_느니르:3'
    self.offset = offset
    note_and_ornament = text.split(':')[0]
    self.note = note_and_ornament.split('_')[0]
    self.ornament = note_and_ornament.split('_')[1:] if '_' in note_and_ornament else None
    self.duration = None
    self.global_offset = None
    if self.note in PITCH2MIDI:
      self.midi = PITCH2MIDI[self.note]
    else:
      self.midi = 0
  
  def __repr__(self) -> str:
    return f'{self.note}({self.midi}) - {self.ornament}, duration:{self.duration} @ {self.offset}, {self.global_offset}'
  
  def __str__(self) -> str:
    ornaments = '_'.join(self.ornament) if self.ornament is not None else ''
    duration = self.duration if self.duration is not None else ''
    output =  f'{self.note}({self.midi}):{ornaments}:{duration}:{self.offset}:{self.global_offset}'
    if len(output.split(':')) != 5: 
      print(f'output: {output}')
    return output
  


class SigimsaeConverter:
  def __init__(self, scale=['황', '태', '중', '임', '남', '무']):
    self.scale = scale
    self.pitch2midi = OrderedDict({scale: PITCH2MIDI[scale] for scale in self.scale})
    octave_name = {"하하배": -3, "하배": -2, "배":-1, '':0, '청':+1, '중청':+2}
    for o in octave_name.keys():
      for p in self.scale:
        self.pitch2midi[o+p] = self.pitch2midi[p] + octave_name[o] * 12
    self.pitch2midi = OrderedDict(sorted(self.pitch2midi.items(), key=lambda x: x[1]))
    self.pitch2up =  {p: list(self.pitch2midi.keys())[i+1] for i, p in enumerate(list(self.pitch2midi.keys())[:-1])}
    self.pitch2down = {p: list(self.pitch2midi.keys())[i-1] for i, p in enumerate(list(self.pitch2midi.keys())[1:])}
    self.midi_scales = list(self.pitch2midi.values())
  
  def up(self, pitch):
    if type(pitch) == int:
      if pitch in self.midi_scales:
        return self.midi_scales[self.midi_scales.index(pitch)+1]
      else:
        print(f'pitch {pitch} not in midi scales')
        return pitch
    return self.pitch2up[pitch]
  
  def down(self, pitch):
    if type(pitch) == int:
      if pitch in self.midi_scales:
        return self.midi_scales[self.midi_scales.index(pitch)-1]
      else:
        print(f'pitch {pitch} not in midi scales')
        return pitch
    return self.pitch2down[pitch]
  
  def up2(self, pitch, n=2):
    if type(pitch) == int:
      return self.midi_scales[self.midi_scales.index(pitch)+n]
    for _ in range(n):
      pitch = self.up(pitch)
    return pitch
  
  def down2(self, pitch, n=2):
    if type(pitch) == int:
      return self.midi_scales[self.midi_scales.index(pitch)-n]
    for _ in range(n):
      pitch = self.down(pitch)
    return pitch


class SymbolReader:
  def __init__(self) -> None:
    self.prev_pitch = 0
    self.prev_offset = 0
    self.dur_ratio = 1.5
    self.sigimsae_conv = SigimsaeConverter()
    self.entire_notes = []
  
  def __call__(self, i, symbol):
      # def handle_symbol(self, i, symbol, prev_pitch, prev_offset, dur_ratio, entire_notes):
    current_offset = (i + symbol.offset) * self.dur_ratio
    if symbol.note == '-':
      return
    if symbol.note == '같은음표':
      symbol.midi = self.prev_pitch
    elif symbol.note == '노':
      symbol.midi = self.sigimsae_conv.down(self.prev_pitch)
    elif symbol.note == '니':
      symbol.midi = self.sigimsae_conv.up(self.prev_pitch)
    elif symbol.note =="로":
      symbol.midi = self.sigimsae_conv.down2(self.prev_pitch)
    elif symbol.note == "리":
      symbol.midi = self.sigimsae_conv.up2(self.prev_pitch)
    elif symbol.note == '니나':
      pass
      # if prev_offset != current_offset:
    if symbol.midi != 0:
      if self.prev_pitch != 0:
        new_note = mnote.Note(self.prev_pitch, quarterLength=current_offset - self.prev_offset)
        self.entire_notes.append(new_note)
      self.prev_offset = current_offset
      self.prev_pitch = symbol.midi
      if symbol.ornament is not None:
        if '니레' in symbol.ornament:
          grace_pitch = self.sigimsae_conv.up(symbol.midi)
          self.entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())
        if '니나' in symbol.ornament:
          grace_pitch = self.sigimsae_conv.up2(symbol.midi)
          self.entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())
        if '노네' in symbol.ornament:
          grace_pitch = self.sigimsae_conv.down(symbol.midi)
          self.entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())
        if '노니로' in symbol.ornament:
          self.entire_notes.append(mnote.Note(symbol.midi, quarterLength=0.5).getGrace())
          grace_pitch = self.sigimsae_conv.up(symbol.midi)
          self.entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())

  def handle_remaining_note(self, i):
    current_offset = i * self.dur_ratio
    new_note = mnote.Note(self.prev_pitch, quarterLength=current_offset - self.prev_offset)
    self.entire_notes.append(new_note)

class JeongganboParser:
  def __init__(self) -> None:
    self.beat_template = self.define_beat_template()
    self.error_text_templates = self.define_error_text_templates()

    self.num_jg_in_gak = 20
    self.num_sharp = -4
    pass
  
  @staticmethod
  def define_beat_template():
    template = {}
    template[(5,)] = [0]
    template[(10, 11)] = [0, Fraction(1,2)]
    template[(10, 14, 15)] = [0, Fraction(1,2), Fraction(3,4)]
    template[(12, 13, 11)] = [0, Fraction(1,4), Fraction(1,2)]
    template[(12, 13, 14, 15)] = [0, Fraction(1,4), Fraction(1,2), Fraction(3,4)]
    template[(2, 5, 8)] = [0, Fraction(1,3), Fraction(2,3)]
    template[(2, 5, 7, 9)] = [0, Fraction(1,3), Fraction(2,3), Fraction(5,6)]
    template[(2, 4, 6, 8)] = [0, Fraction(1,3), Fraction(1,2), Fraction(2,3)]
    template[(2, 4, 6, 7, 9)] = [0, Fraction(1,3), Fraction(1,2), Fraction(2,3), Fraction(5,6)]
    template[(1, 3, 5, 8)] = [0, Fraction(1,6), Fraction(1,3), Fraction(2,3)]
    template[(1, 3, 5, 7, 9)] = [0, Fraction(1,6), Fraction(1,3), Fraction(2,3), Fraction(5,6)]
    template[(1, 3, 4, 6, 8)] = [0, Fraction(1,6), Fraction(1,3), Fraction(1,2), Fraction(2,3)]
    template[(1, 3, 4, 6, 7, 9)] = [0, Fraction(1,6), Fraction(1,3), Fraction(1,2), Fraction(2,3), Fraction(5,6)]

    return template

  @staticmethod
  def define_error_text_templates():
    '''
    hard-coded error text templates
    '''
    error_text_templates = {}
    error_text_templates['고_니레:2 -:5 니나:8'] = '청태_니레:2 -:5 니나:8'
    return error_text_templates

  @staticmethod
  def get_position_tuple(omr_text:str):
    return tuple([int(y) for y in  re.findall(r':(\d+)', omr_text)])


  def parse_omr_results(self, jeonggan: Jeonggan):
    # text = jeonggan.omr_text.replace(':5:5', ':5').replace(':1:1', ':1') # Hard-coded fix for a specific error
    text = jeonggan.omr_text
    notes = text.split(' ')
    notes = [x for x in notes if len(x)>2]
    text = ' '.join(notes)


    if text in self.error_text_templates:
      text = self.error_text_templates[text]
      notes = text.split(' ')
      notes = [x for x in notes if len(x)>2]
    position_tuple = self.get_position_tuple(text)
    if position_tuple not in self.beat_template:
      print(f'position_tuple {position_tuple} not in template', text, jeonggan)
      jeonggan.symbols = [Symbol('-:5', offset=0)]
      return
    by_note_position = self.beat_template[position_tuple]
    out = [Symbol(notes[i], offset=by_note_position[i]) for i in range(len(notes))]
    jeonggan.symbols = out
    return out


  def piece_to_score(self, piece:Piece):
    sb_reader = SymbolReader()
    for i, jeonggan in enumerate(piece.jeonggans):
      if i == 1920: # In Yeominlak, this is where the tempo changes
        sb_reader.handle_remaining_note(i)
        sb_reader.prev_pitch = 0
        sb_reader.dur_ratio = 3.0
        sb_reader.prev_offset = sb_reader.prev_offset * 2
      for symbol in jeonggan.symbols:
        sb_reader(i, symbol)
    sb_reader.handle_remaining_note(i+1)
    score = self.make_score_from_notes(sb_reader.entire_notes)
    return score

  def make_score_from_notes(self, entire_notes: List[mnote.Note]):
    score = stream.Stream()
    score.append(mmeter.TimeSignature(f'{int(self.num_jg_in_gak*3)}/8'))
    current_key = mkey.KeySignature(self.num_sharp)
    score.append(current_key)

    for note in entire_notes:
      note.pitch = mpitch.simplifyMultipleEnharmonics(pitches=[note.pitch], keyContext=current_key)[0]
      if note.pitch.accidental.alter == 0:
        note.pitch.accidental = None # delete natural
      score.append(note)
    return score
  


  @staticmethod
  def parse_duration_and_offset(jeonggans: List[Jeonggan]):
    '''
    Parse duration and offset of each symbol in jeonggans
    '''
    prev_offset = 0 
    prev_pitch = 0
    prev_symbol = None
    for i, jeonggan in enumerate(jeonggans):
      for symbol in jeonggan.symbols:
        symbol.global_offset = i + symbol.offset
        if symbol.note == '-':
          continue
        if symbol.note == '같은음표':
          symbol.midi = prev_pitch
        if prev_symbol is not None:
          prev_symbol.duration = symbol.global_offset - prev_offset
        prev_offset = symbol.global_offset
        prev_pitch = symbol.midi
        prev_symbol = symbol
    prev_symbol.duration = len(jeonggans) - prev_offset
    return None

def piece_to_txt(piece):
  symbols_in_gaks = [','.join([str(symbol) for jeonggan in gak.jeonggans for symbol in jeonggan.symbols]) for gak in piece.gaks if not gak.is_jangdan]
  symbols_in_gaks = '\n'.join(symbols_in_gaks)
  return symbols_in_gaks


def main():
  args = get_parser().parse_args()
  
  reader = JeongganboReader(run_omr=True, omr_model_path=args.model_path)
  parser = JeongganboParser()

  jgb_dir = Path(args.jgb_dir)
  out_dir = Path(args.output_dir)
  low_conf_dir = Path(args.error_dir)
  out_dir.mkdir(exist_ok=True, parents=True)
  low_conf_dir.mkdir(exist_ok=True, parents=True)

  page_by_inst = {'haegeum': (21, 38),
                  'ajaeng': (15,32),
                  'daegeum': (19,36),
                  'gayageum': (21,38),
                  'geomungo': (17,34),
                  'piri': (23,40)
                  }

  piece_by_inst = {}


  for inst, (start, end) in page_by_inst.items():
    target_img_list = sorted([x for x in jgb_dir.glob('*.png') if inst in x.name])[start:end]
    piece_by_inst[inst] = reader.parse_multiple_pages(target_img_list)[0]
  
  
  # Save the parsed results in label format
  for inst, piece in piece_by_inst.items():
    omr_texts = '\n'.join(['|'.join([jg.omr_text for jg in gak.jeonggans]) for gak in piece.gaks if not gak.is_jangdan])
    with open(out_dir/f'{inst}_omr.txt', 'w') as f:
      f.write(omr_texts)


  for inst, piece in piece_by_inst.items():
    print(inst)
    for jeonggan in piece.jeonggans:
      out = parser.parse_omr_results(jeonggan)
    parser.parse_duration_and_offset(piece.jeonggans)


  # piece = piece_by_inst['haegeum']
  entire_score = stream.Score()
  inst_names_in_order = ['daegeum', 'piri', 'haegeum', 'ajaeng', 'gayageum', 'geomungo']
  # inst_names_in_order = ['haegeum', 'ajaeng']
  # inst_names_in_order = ['haegeum']
  total_text = []
  for inst in inst_names_in_order:
    piece = piece_by_inst[inst]
    score = parser.piece_to_score(piece)
    piece_in_text = piece_to_txt(piece)
    total_text.append(piece_in_text)
    entire_score.insert(0, score)
  with open(out_dir/'omr_symbol.txt', 'w') as f:
    f.write('\n\n'.join(total_text))
  entire_score.write('musicxml', fp=str(out_dir/'omr_score.musicxml'))

if __name__ == '__main__':
  main()