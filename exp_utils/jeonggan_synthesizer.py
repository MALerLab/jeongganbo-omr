import re
from random import randint, choice, uniform
from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np

from .const import NAME_EN_TO_KR, NAME_KR_TO_EN, NOTE_W_DUR_EN_SET, SYMBOL_W_DUR_EN_LIST, SYMBOL_WO_DUR_EN_LIST, SYMBOL_WO_DUR_ADD_EN_LIST

INIT_WIDTH = 100
WIDTH_NOISE_SIG = 3.34
WIDTH_NOISE_MIN = -10
WIDTH_NOISE_MAX = 13

INIT_RATIO = 1.4
RATIO_NOISE_SIG = 0.3
RATIO_NOISE_MIN = -0.7
RATIO_NOISE_MAX = 1.0

DEFAULT_MARGIN = 6

MARK_HEIGHT = 26
MARK_WIDTHS = {
  'conti': 21,
  'pause': 27 # 34
}

OCTAVE_WIDTH = 13
OCTAVE_RANGE = 6

# PITCH_ORDER: 'hwang', 'dae', 'tae', 'hyeop', 'go', 'joong', 'yoo', 'lim', 'ee', 'nam', 'mu', 'eung'
PITCH_ORDER = [
                                                                      'lim_ddd',  None,     None,       None,    None,
  'hwang_dd', None, 'tae_dd', 'hyeop_dd',  None,   'joong_dd', None,  'lim_dd',  'ee_dd',  'nam_dd',   'mu_dd',  None,
  'hwang_d',  None, 'tae_d',  'hyeop_d',  'go_d',  'joong_d',  None,  'lim_d',   None,     'nam_d',    'mu_d',   'eung_d',
  'hwang',    None, 'tae',    'hyeop',    'go',    'joong',    None,  'lim',     'ee',     'nam',      'mu',     'eung',
  'hwang_u',  None, 'tae_u',  'hyeop_u',  'go_u',  'joong_u',  None,  'lim_u',   None,     'nam_u',    'mu_u',   None,
  'hwang_uu'
]

class JeongganSynthesizer:
  def __init__(self, img_path_dict):
    self.img_path_dict = img_path_dict
    self.img_dict = {}
    
    for key, path in img_path_dict.items():
      if isinstance(path, list):
        imgs = [ cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in path ]
        self.img_dict[key] = imgs
        continue
        
      self.img_dict[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
  
  def __call__(self, range_limit=True, ornaments=True, apply_noise=True, random_symbols=True, layout_elements=True):
    label, *_, jng_img = self.generate_single_data(range_limit=range_limit, ornaments=ornaments, apply_noise=apply_noise, random_symbols=random_symbols, layout_elements=layout_elements)
    
    return label, jng_img
  
  def generate_single_data(self, range_limit=True, ornaments=True, apply_noise=True, random_symbols=True, layout_elements=True):
    img_w, img_h = self.get_size()
    
    _, label = self.get_label_dict(range_limit=range_limit, ornaments=ornaments)
    
    jng_img = self.generate_image_by_label(label, img_w, img_h, apply_noise=apply_noise, random_symbols=random_symbols, layout_elements=layout_elements)
    
    return label, img_w, img_h, jng_img
  
  # totally empty jng
  def generate_blank_data(self):
    img_w, img_h = self.get_size()
    jng_img = self.get_blank(img_w, img_h)
    jng_dict, label = (
      {
        'row_div': 1,
        'rows': [
          { 
            'col_div': 1,
            'cols': [ '0' ]
          }
        ]
      },
      '0:5'
    )
    
    return label, img_w, img_h, jng_img
  
  # label generator
  @classmethod
  def get_label_dict(cls, div=None, range_limit=True, ornaments=True):
    pitch_range = cls.get_pitch_range()
    
    if not range_limit:
      pitch_range = list(filter(None, PITCH_ORDER))
    
    jng_dict = cls.get_jng_dict( pitch_range, ornaments=ornaments )
    
    label = cls.dict2label(jng_dict)
    
    return jng_dict, label
    
  @staticmethod
  def get_pitch_range():
    num_pitch = len(PITCH_ORDER)
    
    center = randint(0, num_pitch - 1)
    offset = OCTAVE_WIDTH // 2
    
    if center < offset:
      res = PITCH_ORDER[:OCTAVE_WIDTH]
      
    elif center > num_pitch - offset:
      res = PITCH_ORDER[num_pitch-OCTAVE_WIDTH:]
    
    else:
      res = PITCH_ORDER[center-offset:center+offset+1]
    
    return list(filter(None, res))

  @staticmethod
  def get_jng_dict(plist, div=None, ornaments=True):
    if uniform(0, 1) > 0.98:
      return {'row_div': 1, 'rows': [{'col_div': 1, 'cols': ['conti']}]} # 2% chance empty jng
    p_prob = [1] * len(plist) + [5, 1] + [0.2] * len(SYMBOL_W_DUR_EN_LIST)
    p_prob = np.asarray(p_prob) / sum(p_prob)
    plist = plist + ['conti', 'pause'] + SYMBOL_W_DUR_EN_LIST
    sym_set = set(SYMBOL_WO_DUR_EN_LIST)
    
    row_div = div if div else np.random.choice([1,2,3], 1, p=[0.45, 0.1, 0.45])[0]
    
    res = {
      'row_div': row_div,
      'rows': []
    }
    
    for _ in range(row_div):
      col_div = np.random.choice([1,2,3], 1, p=[0.5, 0.49, 0.01])[0] if row_div > 1 else 1
      cols = []
      
      if col_div > 2:
        ornaments = False
      
      for col_idx in range(col_div):
        # group = choice(plist)
        group = np.random.choice(plist, 1, p=p_prob)[0]
        
        if ornaments and group != 'pause' and randint(1, 10) > 6: # add symbol 40% chance
          sym_set_cp = sym_set.copy()
          
          if group in SYMBOL_W_DUR_EN_LIST:
            sym_set_cp -= {'nanina', 'naneuna'}
          
          if col_div > 1 and col_idx < 1:
            sym_set_cp -= { 'flow', 'push' }
          
          sym = choice(list(sym_set_cp))
          
          group = [group] + [sym]
        
        cols.append( group )
      
      res['rows'].append({
        'col_div': col_div,
        'cols': cols
      })
    
    return res
  
  # image generation
  def generate_image_by_label(self, label, width, height, apply_noise=True, random_symbols=True, layout_elements=True):
    jng_dict = self.label2dict(label)
    
    img = self.get_blank(width, height)
    if label=='-:5':
      return self.add_layout_elements(img)
    jng_img = self.generate_image_by_dict(img, jng_dict, apply_noise=apply_noise, random_symbols=random_symbols)
    
    if layout_elements:
      jng_img = self.add_layout_elements(jng_img)
    
    return jng_img
    
  def generate_image_by_dict(self, img, dict, apply_noise=True, random_symbols=True):
    img_h, img_w = img.shape[:2]
    
    jng_arr = [ row['cols'] for row in dict['rows'] ]
    row_div = len(jng_arr)

    jng_infos = []
    row_heights = []

    for row_idx, row in enumerate(jng_arr):
      new_row = []
      row_height = []
      
      for group in row:
        if not isinstance(group, list):
          group = [group]
        
        new_group = []
        
        for el_idx, el_name in enumerate(group):
          el_name_cp = el_name
          
          if el_name in SYMBOL_WO_DUR_ADD_EN_LIST:
            el_name_cp = el_name_cp.split('/')[0]
          
          el_img = self.img_dict[el_name_cp].copy()
          
          if isinstance(el_img, list):
            if apply_noise:
              el_img = choice(el_img).copy()
            else:
              el_img = el_img[0].copy()
          
          el_img_dim = list(el_img.shape[:2])
          
          if el_name in SYMBOL_WO_DUR_ADD_EN_LIST:
            el_img_dim = [ ln*2 for ln in el_img_dim ]
          
          if el_name == 'conti':
            el_img_dim[0] = MARK_HEIGHT
          
          new_group.append( [ el_name, el_img_dim, el_img ])
          row_height.append(el_img_dim[0])
        
        if len(row) < 3 and random_symbols and randint(1, 10) > 7: # 30% chance
          el_name = 'ignore'
          el_img = choice(self.img_dict[el_name]).copy() if bool(randint(0, 1)) else self.make_ignored_symbol()
          el_img_dim = list(el_img.shape[:2])
          
          new_group.append( [ el_name, el_img_dim, el_img ])
          row_height.append(el_img_dim[0])
        
        new_row.append( new_group )
      
      jng_infos.append(new_row)
      row_heights.append(max(row_height))

    row_margin = randint(1, DEFAULT_MARGIN) if apply_noise else 3
    
    if sum(row_heights) > img_h - 2*row_margin:
      ignore = []
      valid = []
      
      for rh in row_heights:
        if rh < 10:
          ignore.append(rh)
        else:
          valid.append(rh * 1.2)
      
      h_size_ratio = (img_h - 2*row_margin - sum(ignore))/sum(valid)
      
      new_jng_infos = []
      new_row_heights = []
      
      for row in jng_infos:
        new_row = []
        new_row_height = []
        
        for group in row:
          new_group = []
          
          for el_name, el_img_dim, el_img in group:
          
            if el_img_dim[0] > 9:
              el_img_dim[0] = int(el_img_dim[0] * h_size_ratio)
            
            if el_name == 'pause':
              el_img_dim[1] = int(el_img_dim[1] * h_size_ratio)
            
            new_group.append([el_name, el_img_dim, el_img])
            new_row_height.append(el_img_dim[0])
          
          new_row.append(new_group)
        
        new_jng_infos.append(new_row)
        new_row_heights.append(max(new_row_height))
      
      jng_infos = new_jng_infos
      row_heights = new_row_heights
      
      del new_jng_infos
      del new_row_heights

    row_gap = int((img_h - 2*row_margin - sum(row_heights)) / (row_div + 1))
    row_template = [ row_margin + sum(row_heights[:idx]) + (idx + 1) * row_gap for idx in range(row_div) ]

    col_margin = 0 # randint(0, DEFAULT_MARGIN)
    
    for row_idx, row in enumerate(jng_infos):
      col_div = len(row)
      row_width = sum([ sum([ el[1][1] for el in group ]) for group in row ])
      
      if row_width > img_w - 2*col_margin:
        ignore = []
        valid = []
        
        for group in row:
          for el_idx, (el_name, el_img_dim, el_img) in enumerate(group):
            # if el_idx > 0:
            #   ignore.append(el_img_dim[1])
            # else:
            valid.append(el_img_dim[1] * 1.1)
        
        w_size_ratio = (img_w - 2*col_margin - sum(ignore))/sum(valid)
        
        new_row = []
        
        for group in row:
          
          new_group = []
          
          for el_idx, (el_name, el_img_dim, el_img) in enumerate(group):
            
            if el_idx < 1: 
              el_img_dim[1] = int(el_img_dim[1] * w_size_ratio)
              
            if el_name == 'pause':
              el_img_dim[0] = int(el_img_dim[0] * w_size_ratio)
            
            new_group.append([el_name, el_img_dim, el_img])
          
          new_row.append(new_group)
        
        row = new_row
        del new_row
      
      notes = [] 
      
      for group in row:
        new_group = []
        
        for el_idx, (el_name, el_img_dim, el_img) in enumerate(group):
          if any([tar != src for tar, src in zip(el_img_dim, el_img.shape[:2])]):
            if el_name == 'conti':
              el_img = self.make_mark(el_img, el_img_dim[0])
            elif el_idx == 0:
              el_img = cv2.resize(el_img, dsize=el_img_dim[-1::-1])
          
          new_group.append([el_img, el_name])
        
        notes.append(new_group)
      
      row_width = sum([ group[0][0].shape[1] for group in notes ])
      col_gap = (img_w - 2*col_margin - row_width) // (col_div + 1)
      col_template = [ col_margin + sum([ group[0][0].shape[1] for group in notes[:idx] ]) + (idx + 1) * col_gap for idx in range(col_div) ]
      
      for col_idx, group in enumerate(notes[:]):
        pos_x = col_template[col_idx]
        note_width = group[0][0].shape[1]
        sym_width = sum([ el[0].shape[1] for el in group[1:] ])
        
        group_width = note_width + sym_width
        
        # left positions
        if len(notes) > 1 and col_idx < 1 and pos_x - sym_width < 0:
          if len(notes[col_idx + 1]) < 2 and group_width < col_template[col_idx + 1]:
            col_template[col_idx] = sym_width
          
          else:
            rs_width = int ( (col_template[col_idx + 1] - sym_width) * 0.9 )
            rs_ratio = rs_width / note_width
            
            if 0 < rs_ratio and rs_ratio < 1:
              new_note_img = cv2.resize(group[0][0], dsize=None, fx=rs_ratio, fy=rs_ratio)
              
              new_group = [[new_note_img, group[0][1]], *group[1:]]
              notes[col_idx] = new_group
              
              col_template[col_idx] = sym_width
        
        # right positions
        elif col_idx > 0 and pos_x + group_width > img_w:
          if len(notes[col_idx - 1]) < 2 and col_template[col_idx - 1] + notes[col_idx - 1][0][0].shape[1] + group_width < img_w:
            col_template[col_idx] = col_template[col_idx - 1] + notes[col_idx - 1][0][0].shape[1]
          else:
            rs_width = int((img_w - col_template[col_idx - 1] - sym_width) * 0.8)
            rs_ratio = rs_width / note_width
            
            if 0 < rs_ratio and rs_ratio < 1:
              new_note_img = cv2.resize(group[0][0], dsize=None, fx=rs_ratio, fy=rs_ratio)
              
              new_group = [[new_note_img, group[0][1]], *group[1:]]
              notes[col_idx] = new_group
              col_template[col_idx] = col_template[col_idx - 1] + 1
      
      for col_idx, group in enumerate(notes):
        cur_pos_x = col_template[col_idx]
        
        for el_idx, (el_img, el_name) in enumerate(group):
          if el_name in SYMBOL_WO_DUR_ADD_EN_LIST:
            min_margin = 4
            
            for split_idx, split_name in enumerate(el_name.split('/')):
              el_img = self.img_dict[split_name]
              
              if isinstance(el_img, list):
                el_img = choice(el_img)
              
              el_img = el_img.copy()
              
              pos_x = col_template[col_idx]
              pos_y = row_template[row_idx] + row_heights[row_idx]//2 - el_img.shape[0]//2  
              
              if split_idx:
                pos_x += group[0][0].shape[1] + min_margin
              else:
                pos_x -= el_img.shape[1] + min_margin
              
              el_img = self.remove_background(el_img)
              img = self.insert_img(img, el_img, pos_x, pos_y)
            
            cur_pos_x = pos_x
            continue
          
          pos_x = cur_pos_x
          pos_y = row_template[row_idx]
          
          if row_heights[row_idx] != el_img.shape[0]:
            pos_y += row_heights[row_idx]//2 - el_img.shape[0]//2
            
          if el_name in {'flow', 'push'}:
            pos_y += int(group[0][0].shape[0] * 0.4)
          
          if el_idx > 0:
            min_margin = 0
            
            if col_div > 1 and col_idx < 1:
              pos_x -= el_img.shape[1] + min_margin
              
            elif col_div < 2 and el_idx > 1 and pos_x + el_img.shape[1] > img_w:
              rs_width = img_w - pos_x
              rs_ratio = rs_width / el_img.shape[1]
              
              if 0 < rs_ratio and rs_ratio < 1:
                el_img = cv2.resize(el_img, dsize=None, fx=rs_ratio, fy=rs_ratio)
          
          if col_div > 1 and col_idx < 1:
            cur_pos_x = pos_x
          else:
            cur_pos_x = pos_x + el_img.shape[1]
          
          if el_idx > 0 and len(group[1:]) < 2:
            if col_div > 1 and col_idx < 1:
              pos_x = col_template[col_idx] // 2 - el_img.shape[1]//2
            else:
              pos_x += (img_w - pos_x)//2 - el_img.shape[1]//2
            
          if apply_noise:
            pos_x += randint(-2, 2)
            pos_y += randint(-2, 2)
          
          if apply_noise:
            rand_ratio = uniform(0.9, 1.1)
            el_img = self.resize_img_by_height(el_img, round(el_img.shape[0] * rand_ratio))
        
          el_img = self.remove_background(el_img)
          
          if -1 * el_img.shape[1] < pos_x and pos_x < img_w and -1 * el_img.shape[0] < pos_y and pos_y < img_h:
            img = self.insert_img(img, el_img, pos_x, pos_y)
    
    return img
  
  def add_layout_elements(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # add random borders
    border = [ randint(0, 3) for _ in range(4) ]
    img = np.pad(img[border[0]:img.shape[0]-border[1], border[2]:img.shape[1]-border[3]], ((border[0], border[1]), (border[2], border[3])), mode='constant', constant_values=0)
    
    # add random lines
    is_vert = bool(randint(0, 1)) # True: vertical line, False: horizontal line
    is_neg = bool(randint(0, 1))
    line_len = randint(5, 20)
    line_weight = randint(1, 5)
    
    if is_vert:
      line_pos = randint(3, img.shape[1]-line_weight)
      if is_neg:
        img[img.shape[0]-line_len:img.shape[0], line_pos:line_pos+line_weight] = 0
      else:
        img[:line_len, line_pos:line_pos+line_weight] = 0
      
    else:
      line_pos = randint(3, img.shape[0]-line_weight)
      if is_neg:
        img[line_pos:line_pos+line_weight, img.shape[1]-line_len:img.shape[1]] = 0
      else:
        img[line_pos:line_pos+line_weight, :line_len] = 0
  
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    return img
  
  @staticmethod
  def clamp(val, _min, _max):
    return min(_max, max(_min, val))

  @classmethod
  def get_width(cls):
    noise = cls.clamp( round(np.random.normal(0, WIDTH_NOISE_SIG)), WIDTH_NOISE_MIN, WIDTH_NOISE_MAX )
    return INIT_WIDTH + noise

  @classmethod
  def get_ratio(cls, min_ratio=None): 
    min_noise = (min_ratio - INIT_RATIO) if min_ratio else RATIO_NOISE_MIN
    noise = round( cls.clamp( np.random.normal(0, RATIO_NOISE_SIG), min_noise, RATIO_NOISE_MAX ), 1 )
    return INIT_RATIO + noise 

  @classmethod
  def get_size(cls, min_ratio=None):
    width = cls.get_width()
    ratio = cls.get_ratio( min_ratio=min_ratio ) 
    
    return width, round(width * ratio)
  
  @staticmethod
  def remove_background(img, crop=True):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    y, x = np.where(img_grey > 250)
    
    img_cp = img.copy()
    
    img_cp[y, x] = np.full(4, 0, dtype=np.uint8)
    
    return img_cp

  @staticmethod
  def get_blank(width, height, bg='white'):
    blank = np.zeros((height, width, 4), dtype=np.uint8)
    
    if bg == 'white':
      blank[:, :] = np.array([255, 255, 255, 255], dtype=np.uint8)
    elif bg == 'black':
      blank[:, :] = np.array([0, 0, 0, 255], dtype=np.uint8)
    
    return blank
  
  @staticmethod
  def insert_img(src, insert, x, y):
    h, w = insert.shape[:2]
    
    x_start = min(max(0, x), src.shape[1])
    y_start = min(max(0, y), src.shape[0])
    
    x_end = min(src.shape[1], x+w) # clamp
    y_end = min(src.shape[0], y+h) # clamp
    
    x_crop_start = max(-x, 0)
    y_crop_start = max(-y, 0)
    
    x_crop_end = min(src.shape[1]-x, insert.shape[1]) if x < src.shape[1] else 0
    y_crop_end = min(src.shape[0]-y, insert.shape[0]) if y < src.shape[0] else 0
    
    insert = insert[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
    
    # normalize alpha to 0 ~ 1
    src_alpha = src[y_start:y_end, x_start:x_end, 3] / 255.0
    ins_alpha = insert[:, :, 3] / 255.0

    # blend src and insert channel by channel
    for ch in range(0, 3):
      src[y_start:y_end, x_start:x_end, ch] = ins_alpha * insert[:, :, ch] + \
                                  src_alpha * src[y_start:y_end, x_start:x_end, ch] * (1 - ins_alpha)

    # denoramlize alpha to 0 ~ 255
    src[y_start:y_end, x_start:x_end, 3] = (1 - (1 - ins_alpha) * (1 - src_alpha)) * 255
    
    return src

  @staticmethod
  def resize_img_by_height(img, target_height):
    og_h, og_w = img.shape[:2]
    
    resize_ratio = target_height / og_h
    resize_width = round(og_w * resize_ratio)
    
    return cv2.resize(img, dsize=(resize_width, target_height), interpolation=None)
  
  @classmethod
  def make_mark(cls, img, height):
    bg = cls.get_blank(img.shape[1], height)
    return cls.insert_img(bg, img, 0, MARK_HEIGHT//2 - img.shape[0]//2)
  
  def make_ignored_symbol(self):
    sym_h, sym_w = choice(self.img_dict['ignore']).shape[:2]
    sym_img = np.random.randint(0, 255, (sym_h, sym_w, 1), dtype=np.uint8)
    return cv2.cvtColor(sym_img, cv2.COLOR_GRAY2RGBA)
    
  @staticmethod
  def dict2label(result_dict):
    # result_dict: { row_div: int, rows: list } 
    # rows: [ { col_div: int, cols: [ group ] } ]
    # group: str or list
    
    def cvt_name(g):
      result = ''
      
      if isinstance(g, list):
        result = '_'.join([ NAME_EN_TO_KR[el] for el in g ])  
      else:
        result = NAME_EN_TO_KR[g]
      
      return result
    
    result_str = ''
    
    row_div, rows = itemgetter('row_div', 'rows')(result_dict)
    
    if row_div == 1:
      group = rows[0]['cols'][0]
      group = cvt_name(group)
      
      result_str += f'{group}' + ':' + str(5)
    
    elif row_div == 3:
      for row_idx, row in enumerate(rows):
        col_div, cols = itemgetter('col_div', 'cols')(row)
        
        if col_div == 1:
          group = cols[0]
          group = cvt_name(group)
          result_str += group + ':' + str(2 + 3 * row_idx) + ' '
          
        elif col_div == 2:
          for col_idx, group in enumerate(cols):
            group = cvt_name(group)
            result_str += group + ':' + str( 1 + (3 * row_idx) + (2 * col_idx) ) + ' '
            
        elif col_div == 3:
          for col_idx, group in enumerate(cols):
            group = cvt_name(group)
            result_str += group + ':' + str( (3 * row_idx) + (col_idx + 1) ) + ' '
    
    elif row_div == 2:
      for row_idx, row in enumerate(rows):
        col_div, cols = itemgetter('col_div', 'cols')(row)
        
        if col_div == 1:
          group = cols[0]
          group = cvt_name(group)
          result_str += group + ':' + str(10 + row_idx) + ' '
          
        elif col_div == 2:
          for col_idx, group in enumerate(cols):
            group = cvt_name(group)
            result_str += group + ':' + str( 12 + (2*row_idx) + col_idx ) + ' '
        
        elif col_div == 3:
          for col_idx, group in enumerate(cols):
            group = cvt_name(group)
            result_str += group + ':' + str( (1 - col_idx%2) * (12 + col_idx//2 + row_idx * 2) + col_idx%2 * (10 + row_idx) ) + ' '
    
    return result_str.strip()
  
  @classmethod
  def label2dict(cls, label: str):
    pattern = r'([^_\s:]+|_+[^_\s:]+|[^:]\d+|[-])'
    
    notes = label.split()
    
    token_groups = []
    
    for note in notes:
      group = re.findall(pattern, note)
      token_groups.append( group )
    
    row_div = cls.get_row_div( [ int(g[-1]) for g in token_groups ] )
    rows = [ { 'col_div': 0, 'cols': [] } for _ in range(row_div) ]
    
    for group in token_groups:
      *group, pos  = group
      row_idx = cls.pos2colIdx(pos, row_div)
      
      group = [ NAME_KR_TO_EN[el.replace('_', '')] for el in group ]
      
      if len(group) < 2:
        group = group[0]
      
      rows[row_idx]['col_div'] += 1
      rows[row_idx]['cols'].append( group )
    
    return {
      'row_div': row_div,
      'rows': rows
    }
  
  @staticmethod
  def get_row_div(poses):
    row_div = 0
    
    if len(poses) == 1 and poses[0] == 5:
      row_div = 1
      
    elif sum([ 1 if pos >= 10 else 0 for pos in poses ]) == len(poses):
      row_div = 2

    else:
      row_div = 3
    
    return row_div
  
  @staticmethod
  def pos2colIdx(pos, row_div):
    pos = int(pos)
    
    if row_div == 1:
      return 0
    
    elif row_div == 3:
      return (pos - 1) // row_div
    
    else:
      if pos < 12:
        return (pos - 10)
      else:
        return (pos - 12) // row_div
def get_img_paths(img_path_base, sub_dirs):
  if isinstance(img_path_base, str):
    img_path_base = Path(img_path_base)
  
  paths = []
  for sd in sub_dirs:
    paths += list( (img_path_base / sd).glob('*.png') )
  
  raw_dict = {
    str(p).split('/')[-1].replace('.png', ''): str(p) \
    for p in paths
  }

  res_dict = {}

  for name, path in sorted(raw_dict.items(), key=lambda x: x[0]):
    name = re.sub(r'(_\d\d\d)|(_ot)', '', name)
    
    if res_dict.get(name, False):
      res_dict[name].append(path)
    else:
      res_dict[name] = [path]

  for name, paths in res_dict.items():
    if len(paths) < 2:
      res_dict[name] = paths[0]

  return res_dict
