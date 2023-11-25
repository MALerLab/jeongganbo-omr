import cv2
import numpy as np

def make_jeonggan_generator(reader, jngb):
  jngb_gaks_w_jangdan = jngb[0].gaks
  jngb_gaks = list(filter(lambda x: not x.is_jangdan, jngb_gaks_w_jangdan))
  
  for gak in jngb_gaks:
    for jng in gak.jeonggans:
      yield jng.img, reader._process_img(jng.img)

def make_jng_gen_and_list(reader, jngb):
  jng_gen = make_jeonggan_generator(reader, jngb)
  jng_list = list(make_jeonggan_generator(reader, jngb))
  
  return jng_gen, jng_list

def read_jngb(reader, infos):
  name, start, num_page = infos['name'], infos['start'], infos['num_page']
  jngb_paths = [f'pngs/{name}_pg-{str(idx + start).zfill(3)}.png' for idx in range(num_page)]
  jngb = reader.parse_multiple_pages(jngb_paths)
  
  return jngb

def template_matching(img, ptrn, ptrn_size, threshold, mode):
  img_copy = img.copy()
  img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

  ptrn_rs = cv2.resize(ptrn, (ptrn_size, ptrn_size))
  ptrn_gray = cv2.cvtColor(ptrn_rs, cv2.COLOR_BGR2GRAY)

  result = cv2.matchTemplate(img_gray, ptrn_gray, mode)

  yCords, xCords = np.where(result >= threshold) 
  
  return yCords, xCords, result

COLOR_DICT = {
  'hwang_dd': (130, 130, 0),
  'hwang_d': (200, 200, 0),
  'hwang': (230, 230, 0),
  'hwang_u': (255, 255, 90),
  'hwang_uu': (255, 255, 150),

  'joong_dd': (100, 0, 0),
  'joong_d': (200, 0, 0),
  'joong': (255, 0, 0),
  'joong_u': (255, 128, 128),
  'joong_uu': (255, 192, 192),

  'lim_dd': (0, 100, 0), 
  'lim_d': (0, 170, 0), 
  'lim': (0, 230, 0), 
  'lim_u': (100, 255, 100), 
  'lim_uu': (182, 255, 182), 
  
  'mu_dd': (100, 0, 100), 
  'mu_d': (130, 0, 130), 
  'mu': (255, 0, 255), 
  'mu_u': (255, 100, 255), 
  'mu_uu': (255, 182, 255), 
  
  'nam_dd': (0, 0, 150), 
  'nam_d': (0, 0, 200), 
  'nam': (0, 0, 255), 
  'nam_u': (80, 80, 255), 
  'nam_uu': (130, 130, 235), 

  'tae_dd': (110, 60, 0), 
  'tae_d': (200, 132, 0), 
  'tae': (255, 165, 0), 
  'tae_u': (255, 192, 100), 
  'tae_uu': (255, 218, 150), 
}

class JngMatcher:
  def __init__(self, ptrn_size, threshold, mode):
    self.ptrn_size = ptrn_size
    self.threshold = threshold
    self.mode = mode
  
  def __call__(self, img, ptrn, ptrn_size=None, threshold=None, mode=None):
    if not ptrn_size:
      ptrn_size = self.ptrn_size
    if not threshold:
      threshold = self.threshold 
    if not mode:
      mode = self.mode
    
    xCords, yCords, _ = self.match(img, ptrn, ptrn_size, threshold, mode)
    
    bboxs = zip(xCords, yCords)
    bboxs_merged = self.merge_bboxs(bboxs, ptrn_size) 
    
    return bboxs_merged
  
  # prep jng img
  @staticmethod
  def remove_border_add_padding(jng_img):
    return np.pad(jng_img[:-3, :-3], ((6, 6), (6, 6), (0, 0)), mode='constant', constant_values=255)
  
  # template matching
  def match(self, img, ptrn, ptrn_size=None, threshold=None, mode=None):
    if not ptrn_size:
      ptrn_size = self.ptrn_size
    if not threshold:
      threshold = self.threshold 
    if not mode:
      mode = self.mode
    
    yCords, xCords, match_result = self.template_match(img, ptrn, ptrn_size, threshold, mode)
    
    return xCords, yCords, match_result
  
  # template matching for testing
  @staticmethod
  def template_match(img, ptrn, ptrn_size, threshold, mode):
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    ptrn_rs = cv2.resize(ptrn, (ptrn_size, ptrn_size))
    ptrn_gray = cv2.cvtColor(ptrn_rs, cv2.COLOR_BGR2GRAY)

    match_result = cv2.matchTemplate(img_gray, ptrn_gray, mode)

    yCords, xCords = np.where(match_result >= threshold) 
    
    return yCords, xCords, match_result
  
  @staticmethod
  def get_overlap_area(source, target):
    s_tl_x, s_tl_y, s_br_x, s_br_y, *_ = source 
    t_tl_x, t_tl_y, t_br_x, t_br_y, *_ = target
    
    overlap_width = s_br_x - s_tl_x
    overlap_height = s_br_y - s_tl_y
    
    if ( np.array(source[:-1]) == np.array(target[:-1]) ).all():
      return overlap_width * overlap_height
    
    if s_tl_x < t_tl_x < s_br_x:
      overlap_width = s_br_x - t_tl_x
    else:
      overlap_width = t_br_x - s_tl_x
    
    if s_tl_y < t_tl_y < s_br_y:
      overlap_height = s_br_y - t_tl_y
    else:
      overlap_height = t_br_y - s_tl_y
      
    return overlap_width * overlap_height
  
  @classmethod
  def is_overlap(cls, source, target, min_ratio=0.2):
    # tl: top-left / br: bottom-right
    s_tl_x, s_tl_y, s_br_x, s_br_y, *_ = source 
    t_tl_x, t_tl_y, t_br_x, t_br_y, *_ = target
    
    # source top-left x >= target bottom-right x OR target top-left x >= source bottom-right x
    if ( s_tl_x >= t_br_x or t_tl_x >= s_br_x ):
      return False
    # source top-left y >= target bottom-right y OR target top-left y >= source bottom-right y
    if ( s_tl_y >= t_br_y or t_tl_y >= s_br_y ):
      return False
    
    source_area = (s_br_x - s_tl_x) * (s_br_y - s_tl_y)
    overlap_area = cls.get_overlap_area(source, target)
    overlap_ratio = overlap_area / source_area
    
    return overlap_ratio > min_ratio

  @classmethod
  def get_overlaps(cls, boxs, target_box, target_idx, min_ratio=0.2):
    overlaps = []
    for idx, box in enumerate(boxs):
      # if idx != target_idx and cls.is_overlap(target_box, box, min_ratio):
      if idx > target_idx and cls.is_overlap(target_box, box, min_ratio):
        overlaps.append( (box, idx) )
    return overlaps

  #AVERAGE CENTER POINT AND MAX CONFIDENCE
  @staticmethod
  def merge_boxs(boxs):
    centers_x = []
    centers_y = []
    
    for box in boxs:
      tl_x, tl_y, br_x, br_y, _ = box
      center_x = tl_x + (br_x - tl_x)//2
      center_y = tl_y + (br_y - tl_y)//2
      
      centers_x.append(center_x)
      centers_y.append(center_y)
    
    w_half = (boxs[0][2] - boxs[0][0])//2
    h_half = (boxs[0][3] - boxs[0][1])//2
    
    center_x = round(np.mean(centers_x))
    center_y = round(np.mean(centers_y))
    
    *_, max_confi =  max(boxs, key=lambda b: b[3])
    
    return center_x - w_half, center_y - h_half, center_x + w_half, center_y + h_half, max_confi
  
  # FILTER BY MAX CONFIDENCE
  # @staticmethod
  # def merge_boxs(boxs):
  #   return max(boxs, key=lambda b: b[3])

  def merge_bboxs(self, bboxs, ptrn_size=None, min_ratio=0.2):
    if not ptrn_size:
      ptrn_size = self.ptrn_size
    
    bboxs = [(bbox[0], bbox[1], bbox[0] + ptrn_size, bbox[1] + ptrn_size, bbox[2]) for bbox in bboxs]
    bboxs = sorted(bboxs, key=lambda b: b[1])
    
    bboxs_merged = []
    index = 0

    while True:
      if index > len(bboxs) - 1:
        break
      
      curr = bboxs[index]

      # overlapping box indexs
      overlaps = self.get_overlaps(bboxs, curr, index, min_ratio)
      
      if len(overlaps) > 0:
        overlaps_box = [ tup[0] for tup in overlaps]
        overlaps_box.append(curr)
        
        bboxs_merged.append( self.merge_boxs(overlaps_box) )
        
        # remove overlaps
        overlaps_idx = [ tup[1] for tup in overlaps ]
        overlaps_idx.sort(reverse=True)
        for overlap_idx in overlaps_idx:
          assert overlap_idx < len(bboxs), f'index out of range: {overlap_idx} / {len(bboxs) - 1}' 
          del bboxs[overlap_idx]
          
      else:
        bboxs_merged.append(curr)
    
      index += 1
    
    return bboxs_merged