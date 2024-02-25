from random import randint, choice, uniform

import cv2
import numpy as np

from exp_utils import JeongganProcessor

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
  
  def __call__(self):
    img_w, img_h = self.get_size()
    img = self.get_blank(img_w, img_h)
    jng_dict, label = self.get_label_dict()
    
    jng_img = self.generate_image_by_dict(img, jng_dict)
    
    return label, jng_img
  
  def generate_single_data(self, range_limit=True):
    img_w, img_h = self.get_size()
    img = self.get_blank(img_w, img_h)
    
    jng_dict, label = self.get_label_dict(range_limit=range_limit)
    
    jng_img = self.generate_image_by_dict(img, jng_dict)
    
    return label, img_w, img_h
  
  # label generator
  @classmethod
  def get_label_dict(cls, div=None, range_limit=True):
    pitch_range = cls.get_pitch_range()
    
    if not range_limit:
      pitch_range = list(filter(None, PITCH_ORDER))
    
    jng_dict = cls.get_jng_dict( pitch_range )
    
    label = JeongganProcessor.get_label(jng_dict)
    
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
  def get_jng_dict(plist, div=None):
    plist = ['conti', 'pause'] + plist
    
    row_div = div if div else randint(1, 3)
    
    res = {
      'row_div': row_div,
      'rows': []
    }
    
    for _ in range(row_div):
      col_div = randint(1, 2) if row_div > 1 else 1
      cols = []
      
      for _ in range(col_div):
        cols.append( choice(plist) )
      
      res['rows'].append({
        'col_div': col_div,
        'cols': cols
      })
    
    return res
  
  # image generation
  def generate_image_by_label(self, label, width, height):
    jng_dict = JeongganProcessor.label2dict(label)
    
    img = self.get_blank(width, height)
    
    jng_img = self.generate_image_by_dict(img, jng_dict)
    
    return jng_img
    
  def generate_image_by_dict(self, img, dict, apply_noise=True):
    img_h, img_w = img.shape[:2]
    
    jng_arr = [ row['cols'] for row in dict['rows'] ]
    row_div = len(jng_arr)

    jng_infos = []
    row_heights = []

    for row_idx, row in enumerate(jng_arr):
      new_row = []
      row_height = []
      
      for note_name in row:
        note_img_path = self.img_path_dict[note_name]
        
        if isinstance(note_img_path, list):
          note_img_path = choice(note_img_path)
        
        note_img = cv2.imread(note_img_path, cv2.IMREAD_UNCHANGED)
        note_img_dim = list(note_img.shape[:2])
        
        if note_name == 'conti' and len(row) == 1:
          note_img_dim[0] = MARK_HEIGHT
        
        new_row.append( [note_name, note_img_dim, note_img] )
        row_height.append(note_img_dim[0])
      
      jng_infos.append(new_row)
      row_heights.append(max(row_height))

    row_margin = randint(1, DEFAULT_MARGIN)
    
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
        
        for col in row:
          note_name, note_img_dim, note_img = col
          
          if note_img_dim[0] > 9:
            note_img_dim[0] = int(note_img_dim[0] * h_size_ratio)
          
          if note_name == 'pause':
            note_img_dim[1] = int(note_img_dim[1] * h_size_ratio)
          
          new_row.append([note_name, note_img_dim, note_img])
          new_row_height.append(note_img_dim[0])
        
        new_jng_infos.append(new_row)
        new_row_heights.append(max(new_row_height))
      
      jng_infos = new_jng_infos
      row_heights = new_row_heights
      
      del new_jng_infos
      del new_row_heights

    row_gap = int((img_h - 2*row_margin - sum(row_heights)) / (row_div + 1))
    row_template = [ row_margin + sum(row_heights[:idx]) + (idx + 1) * row_gap for idx in range(row_div) ]

    col_margin = randint(0, DEFAULT_MARGIN)
    
    for row_idx, row in enumerate(jng_infos):
      col_div = len(row)
      
      if sum([ img_dim[1] for _, img_dim, _ in row ]) > img_w - 2*row_margin:
        ignore = []
        valid = []
        
        for note_name, note_img_dim, _ in row:
          if note_name == 'conti':
            ignore.append(note_img_dim[1])
          else:
            valid.append(note_img_dim[1] * 1.2)
        
        w_size_ratio = (img_w - 2*col_margin - sum(ignore))/sum(valid)
        
        new_row = []
        
        for note_name, note_img_dim, note_img in row:
          if note_name != 'conti':
            note_img_dim[1] = int(note_img_dim[1] * w_size_ratio)
          
          if note_name == 'pause':
            note_img_dim[0] = int(note_img_dim[0] * w_size_ratio)
          
          new_row.append([note_name, note_img_dim, note_img])
        
        row = new_row
        del new_row
      
      notes = [] 
      for note_name, note_img_dim, note_img in row:
        if any([tar != src for tar, src in zip(note_img_dim, note_img.shape[:2])]):
          if note_name == 'conti':
            note_img = self.make_mark(note_img, note_img_dim[0])
          else:
            note_img = cv2.resize(note_img, dsize=note_img_dim[-1::-1])
        
        notes.append((note_img, note_name))
      
      row_width = sum([ img.shape[1] for img, _ in notes ])
      col_gap = (img_w - 2*col_margin - row_width) // (col_div + 1)
      col_template = [ col_margin + sum([ img.shape[1] for img, _ in notes ][:idx]) + (idx + 1) * col_gap for idx in range(col_div) ]
      
      for col_idx, note_tuple in enumerate(notes):
        note_img, note_name = note_tuple
        
        # size noise
        if apply_noise:
          rand_ratio = uniform(0.9, 1.1)
          note_img = self.resize_img_by_height(note_img, round(note_img.shape[0] * rand_ratio))
        
        pos_x = col_template[col_idx]
        pos_y = row_template[row_idx]
        
        if row_heights[row_idx] != note_img.shape[0]:
          pos_y += row_heights[row_idx]//2 - note_img.shape[0]//2
        
        if apply_noise:
          pos_x += randint(-2, 2)
          pos_y += randint(-2, 2)
      
        note_img = self.remove_background(note_img)
      
        img = self.insert_img(img, note_img, pos_x, pos_y)
    
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
    
    x = abs(x)
    y = abs(y)
    
    x_end = min(src.shape[1], x+w) # clamp
    y_end = min(src.shape[0], y+h) # clamp
    
    # print(x, y, w, h, x_end, y_end, src.shape[1::-1])
    # print(src[y:y+h, x:x+w].shape, src[y:y_end, x:x_end].shape)
    # print()
    
    x_crop = min(src.shape[1]-x, insert.shape[1])
    y_crop = min(src.shape[0]-y, insert.shape[0])
    
    insert = insert[:y_crop, :x_crop]
    
    # normalize alpha to 0 ~ 1
    src_alpha = src[y:y_end, x:x_end, 3] / 255.0
    ins_alpha = insert[:, :, 3] / 255.0

    # blend src and insert channel by channel
    for ch in range(0, 3):
      src[y:y_end, x:x_end, ch] = ins_alpha * insert[:, :, ch] + \
                                  src_alpha * src[y:y_end, x:x_end, ch] * (1 - ins_alpha)

    # denoramlize alpha to 0 ~ 255
    src[y:y_end, x:x_end, 3] = (1 - (1 - ins_alpha) * (1 - src_alpha)) * 255
    
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