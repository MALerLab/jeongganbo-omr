class GencodeConverter:
  @staticmethod
  def split_line_to_jg(line):
    jgs = line.split('|')
    cleaned_jgs = []
    for jg in jgs:
      if 'OR' in jg:
        cleaned_jgs.append(jg.split('OR')[1])
      else:
        cleaned_jgs.append(jg)
    return cleaned_jgs

  @staticmethod
  def split_jg_to_notes(jg):
    notes = jg.split(' ')
    notes = [note for note in notes if len(note) > 0]
    return notes

  @classmethod
  def convert_jg_to_gencode(cls, jg):
    notes = cls.split_jg_to_notes(jg)
    pitches = [x.split(':')[0] for x in notes]
    positions = [x.split(':')[1] for x in notes]
    if set(pitches) == {'-'}:
      return ''
    if len(positions) == 1:
      positions[0] = '0'
    outputs = []
    for pos, pitch in zip(positions, pitches):
      if pitch == '-':
        continue
      outputs.append(f":{pos} {pitch.replace('_', ' ')}")
    return ' '.join(outputs)

  @classmethod
  def convert_line_to_gencode(cls, line):
    return ' | '.join([cls.convert_jg_to_gencode(jg) for jg in cls.split_line_to_jg(line)])
    
  @classmethod
  def convert_lines_to_gencode(cls, lines):
    return ' \n '.join([cls.convert_line_to_gencode(line) for line in lines])
  
  @classmethod
  def convert_txt_to_gencode(cls, txt_fn, multi_inst=False):
    with open(txt_fn, 'r') as f:
      lines = f.read()
    if multi_inst:
      insts = lines.split('\n\n')
      insts = [inst for inst in insts if len(inst)>0]
      return '\n\n'.join([cls.convert_lines_to_gencode(inst.split('\n')) for inst in insts])
    lines = lines.split('\n')
    return cls.convert_lines_to_gencode(lines)

  def reverse_convert(self, gencode):
    return


def main(args):
  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(exist_ok=True, parents=True)

  for txt_fn in input_dir.glob('*.txt'):
    try: 
      converted_text = GencodeConverter.convert_txt_to_gencode(txt_fn, multi_inst=True)
      with open(output_dir/txt_fn.name, 'w') as f:
        f.write(converted_text)
    except:
      print(f"Error occured at {txt_fn}")
      continue
  return

if __name__ == '__main__': 
  import argparse
  from pathlib import Path
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  
  args = parser.parse_args()

    
  main(args)