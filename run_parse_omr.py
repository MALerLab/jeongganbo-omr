import re
import pickle
from fractions import Fraction
from pathlib import Path
from collections import OrderedDict
from data_utils import JeongganboReader, Jeonggan
from music21 import stream, note as mnote, meter as mmeter, key as mkey, pitch as mpitch

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


def define_error_text_templates():
  '''
  hard-coded error text templates
  '''
  error_text_templates = {}
  error_text_templates['고_니레:2 -:5 니나:8'] = '청태_니레:2 -:5 니나:8'
  return error_text_templates


def get_position_tuple(omr_text:str):
  return tuple([int(y) for y in  re.findall(r':(\d+)', omr_text)])


def parse_omr_results(jeonggan: Jeonggan):
  # text = jeonggan.omr_text.replace(':5:5', ':5').replace(':1:1', ':1') # Hard-coded fix for a specific error
  text = jeonggan.omr_text
  notes = text.split(' ')
  notes = [x for x in notes if len(x)>2]
  text = ' '.join(notes)

  beat_template = define_beat_template()
  error_text_templates = define_error_text_templates()

  if text in error_text_templates:
    text = error_text_templates[text]
    notes = text.split(' ')
    notes = [x for x in notes if len(x)>2]
  position_tuple = get_position_tuple(text)
  if position_tuple not in beat_template:
    print(f'position_tuple {position_tuple} not in template', text, jeonggan)
    return
  by_note_position = beat_template[position_tuple]
  out = [Symbol(notes[i], offset=by_note_position[i]) for i in range(len(notes))]
  jeonggan.symbols = out
  return out

def piece_to_score(piece):
  current_offset = 0
  entire_notes = []
  prev_pitch = 0
  prev_offset = 0
  dur_ratio = 1.5
  sigimsae_conv = SigimsaeConverter()

  for i, jeonggan in enumerate(piece.jeonggans):
    if i == 1920: # In Yeominlak, this is where the tempo changes
      current_offset = i * dur_ratio
      new_note = mnote.Note(prev_pitch, quarterLength=current_offset - prev_offset)
      entire_notes.append(new_note)
      prev_pitch = 0
      dur_ratio = 3.0
      prev_offset = prev_offset * 2
    for symbol in jeonggan.symbols:
      current_offset = (i + symbol.offset) * dur_ratio
      if symbol.note == '-':
        continue
      if symbol.note == '같은음표':
        symbol.midi = prev_pitch
      elif symbol.note == '노':
        symbol.midi = sigimsae_conv.down(prev_pitch)
      elif symbol.note == '니':
        symbol.midi = sigimsae_conv.up(prev_pitch)
      elif symbol.note =="로":
        symbol.midi = sigimsae_conv.down2(prev_pitch)
      elif symbol.note == "리":
        symbol.midi = sigimsae_conv.up2(prev_pitch)
      elif symbol.note == '니나':
        pass
        # if prev_offset != current_offset:
      if symbol.midi != 0:
        if prev_pitch != 0:
          new_note = mnote.Note(prev_pitch, quarterLength=current_offset - prev_offset)
          entire_notes.append(new_note)
        prev_offset = current_offset
        prev_pitch = symbol.midi
        if symbol.ornament is not None:
          if '니레' in symbol.ornament:
            grace_pitch = sigimsae_conv.up(symbol.midi)
            entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())
          if '니나' in symbol.ornament:
            grace_pitch = sigimsae_conv.up2(symbol.midi)
            entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())
          if '노네' in symbol.ornament:
            grace_pitch = sigimsae_conv.down(symbol.midi)
            entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())
          if '노니로' in symbol.ornament:
            entire_notes.append(mnote.Note(symbol.midi, quarterLength=0.5).getGrace())
            grace_pitch = sigimsae_conv.up(symbol.midi)
            entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())

  current_offset = (i+1) * dur_ratio
  new_note = mnote.Note(prev_pitch, quarterLength=current_offset - prev_offset)
  entire_notes.append(new_note)

  score = stream.Stream()
  score.append(mmeter.TimeSignature('60/8'))
  current_key = mkey.KeySignature(-4)
  score.append(current_key)

  for note in entire_notes:
    note.pitch = mpitch.simplifyMultipleEnharmonics(pitches=[note.pitch], keyContext=current_key)[0]
    if note.pitch.accidental.alter == 0:
      note.pitch.accidental = None # delete natural
    score.append(note)
  return score


def parse_duration_and_offset(jeonggans):
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
  reader = JeongganboReader(run_omr=True)

  jgb_dir = Path('jeongganbo-png/pngs/')

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

  for inst, piece in piece_by_inst.items():
    print(inst)
    for jeonggan in piece.jeonggans:
      out = parse_omr_results(jeonggan)
    parse_duration_and_offset(piece.jeonggans)


  piece = piece_by_inst['haegeum']
  entire_score = stream.Score()
  inst_names_in_order = ['daegeum', 'piri', 'haegeum', 'ajaeng', 'gayageum', 'geomungo']
  # inst_names_in_order = ['haegeum']
  total_text = []
  for inst in inst_names_in_order:
    piece = piece_by_inst[inst]
    score = piece_to_score(piece)
    piece_in_text = piece_to_txt(piece)
    total_text.append(piece_in_text)
    entire_score.insert(0, score)
  with open(f'yeominlak_omr_in_text.txt', 'w') as f:
    f.write('\n\n'.join(total_text))
  entire_score.write('musicxml', fp='yeominlak_omr_test.musicxml')

if __name__ == '__main__':
  main()