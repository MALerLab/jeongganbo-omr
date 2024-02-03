import math
from random import randint, choice
from operator import itemgetter
import re

import cv2
import numpy as np


# only for experiment purpose
def make_jeonggan_generator(reader, jngb):
  jngb_gaks_w_jangdan = jngb[0].gaks
  jngb_gaks = list(filter(lambda x: not x.is_jangdan, jngb_gaks_w_jangdan))
  
  for gak in jngb_gaks:
    for jng in gak.jeonggans:
      yield jng.img

def make_jeonggan_list(reader, jngb):
  jng_gen = make_jeonggan_generator(reader, jngb)
  
  return list(jng_gen)

def read_jeongganbo(reader, infos):
  name, start, num_page = infos['name'], infos['start'], infos['num_page']
  jngb_paths = [f'pngs/{name}_pg-{str(idx + start).zfill(3)}.png' for idx in range(num_page)]
  jngb = reader.parse_multiple_pages(jngb_paths)
  
  return jngb

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

PNAME_LIST = [ '황', '대', '태', '협', '고', '중', '유', '임', '이', '남', '무', '응', '중청황', '청황', '배황', '하배황', '하하배황', '중청대', '청대', '배대', '하배대', '하하배대', '중청태', '청태', '배태', '하배태', '하하배태', '중청협', '청협', '배협', '하배협', '하하배협', '중청고', '청고', '배고', '하배고', '하하배고', '중청중', '청중', '배중', '하배중', '하하배중', '중청유', '청유', '배유', '하배유', '하하배유', '중청임', '청임', '배임', '하배임', '하하배임', '중청이', '청이', '배이', '하배이', '하하배이', '중청남', '청남', '배남', '하배남', '하하배남', '중청무', '청무', '배무', '하배무', '하하배무', '중청응', '청응', '배응', '하배응', '하하배응', '-']

SPECIAL_CHAR_TO_NAME = {
  '^': "니레",
  'ㄷ': "나니로",
  '(': "추성",
  ')': "퇴성",
}

PNAME_EN_TO_KR = {
  'ee_dd': '하배이',
  'ee': '이',
  'eung_d': '배응',
  'eung': '응',
  'go_d': '배고',
  'go': '고',
  'go_u': '청고',
  'hwang_dd': '하배황',
  'hwang_d': '배황',
  'hwang': '황',
  'hwang_u': '청황',
  'hwang_uu': '중청황',
  'hyeop_dd': '하배협',
  'hyeop_d': '배협',
  'hyeop': '협',
  'hyeop_u': '청협',
  'joong_dd': '하배중',
  'joong_d': '배중',
  'joong': '중',
  'joong_u': '청중',
  'lim_ddd': '하하배임',
  'lim_dd': '하배임',
  'lim_d': '배임',
  'lim': '임',
  'lim_u': '청임',
  'mu_dd': '하배무',
  'mu_d': '배무',
  'mu': '무',
  'mu_u': '청무',
  'nam_dd': '하배남',
  'nam_d': '배남',
  'nam': '남',
  'nam_u': '청남',
  'tae_dd': '하배태',
  'tae_d': '배태',
  'tae': '태',
  'tae_u': '청태',
  'conti': '-',
  'pause': '쉼표'
}

PNAME_KR_TO_EN = {
  '하배이': 'ee_dd',
  '이': 'ee',
  '배응': 'eung_d',
  '응': 'eung',
  '배고': 'go_d',
  '고': 'go',
  '청고': 'go_u',
  '하배황': 'hwang_dd',
  '배황': 'hwang_d',
  '황': 'hwang',
  '청황': 'hwang_u',
  '중청황': 'hwang_uu',
  '하배협': 'hyeop_dd',
  '배협': 'hyeop_d',
  '협': 'hyeop',
  '청협': 'hyeop_u',
  '하배중': 'joong_dd',
  '배중': 'joong_d',
  '중': 'joong',
  '청중': 'joong_u',
  '하하배임': 'lim_ddd',
  '하배임': 'lim_dd',
  '배임': 'lim_d',
  '임': 'lim',
  '청임': 'lim_u',
  '하배무': 'mu_dd',
  '배무': 'mu_d',
  '무': 'mu',
  '청무': 'mu_u',
  '하배남': 'nam_dd',
  '배남': 'nam_d',
  '남': 'nam',
  '청남': 'nam_u',
  '하배태': 'tae_dd',
  '배태': 'tae_d',
  '태': 'tae',
  '청태': 'tae_u',
  '-': 'conti',
  '쉼표': 'pause',
}

class JeongganProcessor:
  def __init__(self, ptrn_size, threshold, mode):
    self.ptrn_size = ptrn_size
    self.threshold = threshold
    self.mode = mode
  
  def __call__(self, img, ptrn_dict, ptrn_size=None, threshold=None, mode=None):
    if not ptrn_size:
      ptrn_size = self.ptrn_size
    if not threshold:
      threshold = self.threshold 
    if not mode:
      mode = self.mode
    
    jng_img = self.remove_borders( img )

    row_templates, col_templates = self.get_position_templates(jng_img.shape[:2])

    jng_match_bbox_groups = self.get_match_bbox_groups(jng_img, ptrn_dict)
    jng_cont_bboxs = self.get_contour_bboxs_with_match(jng_img, jng_match_bbox_groups)
    
    jng_cont_bbox_groups = self.group_contour_bboxs_by_ypos(jng_cont_bboxs, max_dist=5)
    jng_row_groups = [ { 'center': self.get_bbox_group_center_y(group), 'group': sorted(group, key=lambda x: x[0]) } for group in jng_cont_bbox_groups ]
    jng_row_groups = sorted(jng_row_groups, key=lambda x: x['center'])
    
    row_div, row_indices = self.get_aligned_row_indices(jng_img, jng_row_groups, row_templates)
    
    jng_aligned_result = { 
      'row_div': row_div,
      'rows': [None] * row_div
    }
    
    for row_idx, jng_row_group_idx in enumerate(row_indices):
      if jng_row_group_idx == None:
        continue
      
      jng_row_group = jng_row_groups[jng_row_group_idx]['group']
      col_div, col_indices = self.get_aligned_col_indices(jng_row_group, col_templates)
      
      jng_aligned_result['rows'][row_idx] = {
        'col_div': col_div,
        'cols': [ jng_row_group[char_idx] if char_idx != None else char_idx for char_idx in col_indices]
      }
    
    jng_aligned_result_format = jng_aligned_result
    
    for row_idx, row in enumerate(jng_aligned_result_format['rows']):
      if not row:
        continue
      
      for col_idx, col in enumerate(row['cols']):
        if not col:
          continue
        
        jng_aligned_result_format['rows'][row_idx]['cols'][col_idx] = col[-1]
    
    label_str = self.get_label(jng_aligned_result_format)
    
    return label_str, jng_aligned_result
  
  ## prep img: remove black borders on sides
  @staticmethod
  def remove_borders(img, border=2):
    return np.pad(img[border:-border, border:-border], ((border, border), (border, border), (0, 0)), mode='constant', constant_values=255)
  
  
  ## template matching logics
  @staticmethod
  def template_match(img, ptrn, ptrn_size, threshold, mode):
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    ptrn_rs = cv2.resize(ptrn, (ptrn_size, ptrn_size))
    ptrn_gray = cv2.cvtColor(ptrn_rs, cv2.COLOR_BGR2GRAY)

    match_result = cv2.matchTemplate(img_gray, ptrn_gray, mode)

    yCords, xCords = np.where(match_result >= threshold) 
    
    return yCords, xCords, match_result
  
  def get_match_bboxs(self, img, ptrn, ptrn_size=None, threshold=None, mode=None):
    if not ptrn_size:
      ptrn_size = self.ptrn_size
    if not threshold:
      threshold = self.threshold 
    if not mode:
      mode = self.mode
    
    yCords, xCords, match_result = self.template_match(img, ptrn, ptrn_size, threshold, mode)
    
    bboxs_w_confi = [(x, y, x+ptrn_size, y+ptrn_size, match_result[y][x]) for x, y in zip(xCords, yCords)]
    
    return bboxs_w_confi
  
  def get_match_bbox_groups(self, img, ptrn_dict):
    bbox_list = [] # list of (tl_x, tl_y, br_x, br_y, confi, ptrn_name)

    for ptrn_key, ptrn_img in ptrn_dict.items():
      # list of (tl_x, tl_y, br_x, br_y, confi)
      bboxs = self.get_match_bboxs(img, ptrn_img) 
      bboxs_merged = self.merge_match_bboxs(bboxs)
      
      if( len(bboxs_merged) > 0 ):
        bbox_list += [bbox + (ptrn_key,) for bbox in bboxs_merged]

    # list of (tl_x, tl_y, br_x, br_y, confi, ptrn_name)
    bbox_groups = self.group_overlap_match_bboxs(bbox_list)
    
    return bbox_groups
  
  ## match bbox merging logics
  @staticmethod
  def get_overlap_area_match_bbox(source, target):
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
  def is_overlap_match_bbox(cls, source, target, min_ratio=0.2):
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
    overlap_area = cls.get_overlap_area_match_bbox(source, target)
    overlap_ratio = overlap_area / source_area
    
    return overlap_ratio > min_ratio

  @classmethod
  def get_overlaps_match_bbox(cls, bboxs, soruce_bbox, soruce_idx, min_ratio=0.2):
    overlaps = []
    for idx, bbox in enumerate(bboxs):
      if idx > soruce_idx and cls.is_overlap_match_bbox(soruce_bbox, bbox, min_ratio):
        overlaps.append( (bbox, idx) )
    return overlaps

  @staticmethod
  def merge_overlap_match_bboxs(bboxs): #AVERAGE CENTER POINT AND MAX CONFIDENCE
    centers_x = []
    centers_y = []
    
    for bbox in bboxs:
      tl_x, tl_y, br_x, br_y, _ = bbox
      center_x = tl_x + (br_x - tl_x)//2
      center_y = tl_y + (br_y - tl_y)//2
      
      centers_x.append(center_x)
      centers_y.append(center_y)
    
    w_half = (bboxs[0][2] - bboxs[0][0])//2
    h_half = (bboxs[0][3] - bboxs[0][1])//2
    
    center_x = round(np.mean(centers_x))
    center_y = round(np.mean(centers_y))
    
    *_, max_confi =  max(bboxs, key=lambda b: b[3])
    
    return center_x - w_half, center_y - h_half, center_x + w_half, center_y + h_half, max_confi

  def merge_match_bboxs(self, bboxs, ptrn_size=None, min_ratio=0.2):
    if not ptrn_size:
      ptrn_size = self.ptrn_size
    
    bboxs = sorted(bboxs, key=lambda b: b[1])
    
    bboxs_merged = []
    index = 0

    while True:
      if index > len(bboxs) - 1:
        break
      
      curr = bboxs[index]

      # overlapping box indexs
      overlaps = self.get_overlaps_match_bbox(bboxs, curr, index, min_ratio)
      
      if len(overlaps) > 0:
        overlap_boxs = [ tup[0] for tup in overlaps]
        overlap_boxs = [curr] + overlap_boxs
        
        bboxs_merged.append( self.merge_overlap_match_bboxs(overlap_boxs) )
        
        # remove overlaps
        overlap_indices = [ tup[1] for tup in overlaps ]
        overlap_indices.sort(reverse=True)
        for overlap_idx in overlap_indices:
          assert overlap_idx < len(bboxs), f'index out of range: {overlap_idx} / {len(bboxs) - 1}' 
          del bboxs[overlap_idx]
          
      else:
        bboxs_merged.append(curr)
    
      index += 1
    
    return bboxs_merged
  
  def group_overlap_match_bboxs(self, bboxs, ptrn_size=None, min_ratio=0.6):
    bboxs = sorted(bboxs, key=lambda b: b[1])

    bbox_groups = []
    index = 0

    while True:
      if index > len(bboxs) - 1:
        break
      
      curr = bboxs[index]

      # overlapping box indexs
      overlaps = self.get_overlaps_match_bbox(bboxs, curr, index, min_ratio=min_ratio)

      if len(overlaps) > 0:
        overlap_boxs = [ tup[0] for tup in overlaps ]
        overlap_boxs = [curr] + overlap_boxs
        
        bbox_groups.append( overlap_boxs )
        
        # remove overlaps
        overlap_indices = [ tup[1] for tup in overlaps ]
        overlap_indices.sort(reverse=True)
        for overlap_idx in overlap_indices:
          assert overlap_idx < len(bboxs), f'index out of range: {overlap_idx} / {len(bboxs) - 1}' 
          del bboxs[overlap_idx]
          
      else:
        bbox_groups.append( [curr] )

      index += 1

    return bbox_groups
  
  ## contour box logics
  @staticmethod
  def get_bbox_of_match_bbox_group(img, match_bbox_group):
    return_tl = [img.shape[1], img.shape[0]]
    return_br = [0, 0]

    for bbox in match_bbox_group:
      tl_x, tl_y, br_x, br_y, *_ = bbox
      
      if tl_x < return_tl[0]:
        return_tl[0] = tl_x
      if tl_y < return_tl[1]:
        return_tl[1] = tl_y
      
      if br_x > return_br[0]:
        return_br[0] = br_x
      if br_y > return_br[1]:
        return_br[1] = br_y
    
    return return_tl[0], return_tl[1], return_br[0], return_br[1]
  
  @classmethod
  def find_contour_bbox_in_sliced_img(cls, img, filter_edge_content=True):
    img_dim = img.shape[:2] # [height, width]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    conts = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = conts[0] if len(conts) == 2 else conts[1]

    conts_filtered = []

    for c in conts:
      x, y, w, h = cv2.boundingRect(c)
      
      tl_x, tl_y, br_x, br_y = (x, y, x+w, y+h)
      
      if filter_edge_content:
        is_edge_content = tl_x == 0 or tl_y == 0 or br_x == img_dim[1] or br_y == img_dim[0]
        is_small = w < (img_dim[1] * 0.3) or h < (img_dim[0] * 0.3)
        if is_edge_content and is_small:
          continue
      
      conts_filtered.append( (tl_x, tl_y, br_x, br_y) )
  
    contour_bbox = cls.get_bbox_of_match_bbox_group(img, conts_filtered)
    
    return contour_bbox, (conts_filtered if filter_edge_content else None)

  def get_contour_bbox(self, img, match_bbox_group):
    slice_tl_x, slice_tl_y, slice_br_x, slice_br_y = self.get_bbox_of_match_bbox_group(img, match_bbox_group)
    img_sliced = img.copy()[slice_tl_y:slice_br_y, slice_tl_x:slice_br_x]
    
    cont_bbox_sliced, *_ = self.find_contour_bbox_in_sliced_img(img_sliced)
    
    cont_bbox = tuple(el + slice_tl_x if idx%2 == 0 else el + slice_tl_y for idx, el in enumerate(cont_bbox_sliced))
    
    return cont_bbox

  def get_contour_bboxs(self, img, match_bbox_groups):
    cont_bbox_list = []

    for match_bbox_group in match_bbox_groups:
      cont_bbox = self.get_contour_bbox(img, match_bbox_group)
      
      cont_bbox_list.append(cont_bbox)
    
    return cont_bbox_list

  def get_contour_bboxs_with_match(self, img, match_bbox_groups):
    cont_bbox_list = []
    
    # get contour bbox with 
    for match_bbox_group_idx, match_bbox_group in enumerate(match_bbox_groups):
      cont_bbox = self.get_contour_bbox(img, match_bbox_group)
      
      match_filtered = self.filter_match_by_distance(match_bbox_group, cont_bbox)
      
      cont_bbox_list.append( cont_bbox + (match_filtered,) )
    
    return cont_bbox_list
  
  @classmethod
  def get_distance_bbox_centers(cls, source, target, norm='L2', rd=False):
    s_x, s_y = cls.get_bbox_center(source)
    t_x, t_y = cls.get_bbox_center(target)
    
    if norm == 'L1':
      return abs(s_x - t_x) + abs(s_y - t_y)
    
    if norm == 'L2':
      result = math.sqrt( abs(s_x - t_x) ** 2 +  abs(s_y - t_y) ** 2 )
      
      if rd:
        return round(result)
      
      return result

  @classmethod
  def filter_match_by_distance(cls, match_bbox_group, cont_bbox):
    min_dist = math.inf
    dist_list = []
    
    for match_idx, match_bbox in enumerate(match_bbox_group):
      dist = cls.get_distance_bbox_centers(cont_bbox, match_bbox)
      dist_list.append( (dist, match_idx) )
      
      if dist < min_dist:
        min_dist = dist
    
    result_candidates = []
    
    for dist, match_idx in dist_list:
      if dist <= min_dist:
        result_candidates.append(match_bbox_group[match_idx])
    
    result = max(result_candidates, key=lambda x: x[-2])
    
    # match_bbox_group: list of (tl_x, tl_y, br_x, br_y, confi, ptrn_name)
    return result[-1]

  ## contour box grouping logics
  @staticmethod
  def get_bbox_center(box):
    tl_x, tl_y, br_x, br_y, *_ = box
    return tl_x + (br_x - tl_x)//2, tl_y + (br_y - tl_y)//2

  @classmethod
  def is_y_approximate_contour_bbox(cls, target, compare, max_dist=5):  
    _, t_center_y = cls.get_bbox_center(target)
    _, c_center_y = cls.get_bbox_center(compare)
    
    return abs(t_center_y - c_center_y) < max_dist

  @classmethod
  def get_y_approximates_contour_bbox(cls, boxs, target_box, target_idx, max_dist=5):
    approxs = []
    for idx, box in enumerate(boxs):
      if idx > target_idx and cls.is_y_approximate_contour_bbox(target_box, box, max_dist):
        approxs.append( (box, idx) )
    return approxs

  def group_contour_bboxs_by_ypos(self, bboxs, max_dist=5):
    bboxs = sorted(bboxs, key=lambda b: b[1])

    bbox_groups = []
    index = 0

    while True:
      if index > len(bboxs) - 1:
        break
      
      curr = bboxs[index]

      # overlapping box indexs
      approxs = self.get_y_approximates_contour_bbox(bboxs, curr, index, max_dist=max_dist)

      if len(approxs) > 0:
        approx_boxs = [ tup[0] for tup in approxs ]
        approx_boxs = [curr] + approx_boxs
        
        bbox_groups.append( approx_boxs )
        
        # remove approxs
        approx_indices = [ tup[1] for tup in approxs ]
        
        approx_indices.sort(reverse=True)
        for approx_idx in approx_indices:
          assert approx_idx < len(bboxs), f'index out of range: {approx_idx} / {len(bboxs) - 1}' 
          del bboxs[approx_idx]
          
      else:
        bbox_groups.append( [curr] )

      index += 1

    return bbox_groups
  
  @classmethod
  def get_bbox_group_center_y(cls, bbox_group):
    bbox_center_ys = [ cls.get_bbox_center(bbox)[1] for bbox in bbox_group]
    return round(np.mean(bbox_center_ys))
  
  ## position align logics
  @staticmethod
  def get_position_templates(img_dims):
    img_h, img_w = img_dims
    row_templates = [ tuple(img_h//idx * jdx for jdx in range(1, idx)) for idx in range(2, 5) ]
    col_templates = [ tuple(img_w//idx * jdx for jdx in range(1, idx)) for idx in range(2, 4) ]
    
    return row_templates, col_templates

  @staticmethod
  def check_white_space(img, box, max_pix=300):
    img = img.copy()
    
    img_sliced = np.concatenate((img[:box[1]], img[box[3]:]), axis=0)
    img_sliced = np.where( cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY) > 230, 0, 1 )
    
    return np.sum(img_sliced) < max_pix 

  def get_aligned_row_indices(self, img, groups, templates, max_dist=5):
    fit_result_list = []
    
    for template_idx, template in enumerate(templates):
      fit_result = 0
      row_matching_result = [None] * (template_idx + 1)
      
      if len(groups) > len(template):
        fit_result_list.append( (fit_result, template_idx + 1, row_matching_result) )
        continue
      
      last_group_idx = 0
      
      for row_idx, row_pos in enumerate(template):
        group_idx = 0
        
        while True:
          if group_idx > len(groups) - 1:
            break
          
          if group_idx < last_group_idx:
            group_idx += 1
            continue
          
          curr_group = groups[group_idx]
          group_center = curr_group['center']
          
          is_approx = abs(group_center - row_pos) <= max_dist
          if len(template) > 2 and row_idx in (0, 2):
            is_approx = is_approx or (row_idx == 0 and group_center < row_pos)
            is_approx = is_approx or (row_idx == 2 and group_center > row_pos)
          
          if is_approx:
            fit_result += 1
            row_matching_result[row_idx] = group_idx
            last_group_idx = group_idx
            break
          
          group_idx += 1

      fit_result_list.append( (fit_result, template_idx + 1, row_matching_result) )

    if len(groups) == 1 and fit_result_list[0][0] == fit_result_list[2][0] and self.check_white_space(img, groups[0]['group'][0]):
      return fit_result_list[0][1:]
    
    return max(fit_result_list, key=lambda x: x[0])[1:]

  def get_aligned_col_indices(self, group, templates, max_dist=5):
    fit_result_list = []
    
    for template_idx, template in enumerate(templates):
      fit_result = 0
      col_matching_result = [None]* (template_idx + 1)
      
      last_char_idx = 0
      
      for col_idx, col_pos in enumerate(template):
        char_idx = 0
        
        while True:
          if char_idx > len(group) - 1:
            break
          
          if char_idx < last_char_idx:
            char_idx += 1
            continue
          
          curr_char = group[char_idx]
          char_center = curr_char[0] + (curr_char[2] - curr_char[0])//2
          
          is_approx = abs(char_center - col_pos) <= max_dist
          if len(template) > 1:
            is_approx = is_approx or (col_idx == 0 and char_center < col_pos)
            is_approx = is_approx or (col_idx == 1 and char_center > col_pos)
          
          if is_approx:
            fit_result += 1
            col_matching_result[col_idx] = char_idx
            last_char_idx = char_idx
            break
          
          char_idx += 1
          
      fit_result_list.append( (fit_result, template_idx + 1, col_matching_result) )
      
    return max(fit_result_list, key=lambda x: x[0])[1:]
  
  @staticmethod
  def get_label(result_dict):
    # result_dict: { row_div: int, rows: list } 
    # rows: [ { col_div: int, cols: [ pname ] } ]
    
    result_str = ''
    
    row_div, rows = itemgetter('row_div', 'rows')(result_dict)
    
    if row_div == 1:
      if rows[0] == None or rows[0]['cols'][0] == None:
        return ''
      
      pname = rows[0]['cols'][0]
      return result_str + f'{PNAME_EN_TO_KR[pname]}' + ':' + str(5)
    
    if row_div == 3:
      for row_idx, row in enumerate(rows):
        if row == None:
          continue
        
        col_div, cols = itemgetter('col_div', 'cols')(row)
        
        if col_div == 1 and cols[0]:
          pname = cols[0]
          result_str += PNAME_EN_TO_KR[pname] + ':' + str(2 + 3 * row_idx) + ' '
          
        else:
          for col_idx, pname in enumerate(cols):
            if pname == None:
              continue
            
            result_str += PNAME_EN_TO_KR[pname] + ':' + str( 1 + (3 * row_idx) + (2 * col_idx) ) + ' '
      
      return result_str.strip()
    
    if row_div == 2:
      for row_idx, row in enumerate(rows):
        if row == None:
          continue
        
        col_div, cols = itemgetter('col_div', 'cols')(row)
        
        if col_div == 1 and cols[0]:
          
          pname = cols[0]
          result_str += PNAME_EN_TO_KR[pname] + ':' + str(10 + row_idx) + ' '
          
        else:
          for col_idx, pname in enumerate(cols):
            if pname == None:
              continue
            
            result_str += PNAME_EN_TO_KR[pname] + ':' + str( 12 + (2*row_idx) + col_idx ) + ' '
      
      return result_str.strip()
    
    return result_str.strip()
  
  @classmethod
  def label2dict(cls, label: str):
    pattern = r'([^_\s:]+|_+[^_\s:]+|[^:]\d+|[-])'
    
    notes = label.split()
    
    token_groups = []
    
    for note in notes:
      findings = re.findall(pattern, note)
      token_groups.append( findings )
    
    row_div = cls.get_row_div( list( map(lambda g: int(g[-1]), token_groups) ) )
    rows = [ { 'col_div': 0, 'cols': [] } for _ in range(row_div) ]
    
    for group in token_groups:
      pname, *_, pos  = group
      row_idx = cls.pos2colIdx(pos, row_div)
      
      rows[row_idx]['col_div'] += 1
      rows[row_idx]['cols'].append( PNAME_KR_TO_EN[pname] )
    
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
    

INIT_WIDTH = 100
WIDTH_NOISE_SIG = 3.34
WIDTH_NOISE_MIN = -10
WIDTH_NOISE_MAX = 13

INIT_RATIO = 1.4
RATIO_NOISE_SIG = 0.3
RATIO_NOISE_MIN = -0.7
RATIO_NOISE_MAX = 1.0

MARK_HEIGHT = 27
MARK_WIDTHS = {
  'conti': 21,
  'pause': 27
}

OCTAVE_WIDTH = 10
OCTAVE_RANGE = 6

# PITCH_ORDER: 'hwang', 'dae', 'tae', 'hyeop', 'go', 'joong', 'yoo', 'lim', 'ee', 'nam', 'mu', 'eung'
PITCH_ORDER = [
                                                        'lim_ddd', None,     None,       None,     None,
  'hwang_dd', 'tae_dd', 'hyeop_dd',  None,   'joong_dd', 'lim_dd',  'ee_dd',  'nam_dd',   'mu_dd',  None,
  'hwang_d',  'tae_d',  'hyeop_d',  'go_d',  'joong_d',  'lim_d',   None,     'nam_d',    'mu_d',   'eung_d',
  'hwang',    'tae',    'hyeop',    'go',    'joong',    'lim',     'ee',     'nam',      'mu',     'eung',
  'hwang_u',  'tae_u',  'hyeop_u',  'go_u',  'joong_u',  'lim_u',   None,     'nam_u',    'mu_u',   None,
  'hwang_uu'
]

class JeongganSynthesizer:
  def __init__(self, img_path_dict):
    self.img_path_dict = img_path_dict
  
  def __call__(self):
    img_w, img_h = self.get_size()
    img = self.get_blank(img_w, img_h)
    
    jng_dict, label = self.get_label_dict(img_aspect_ratio = img_h/img_w)
    
    jng_img = self.generate_image_by_dict(img, jng_dict)
    
    return label, jng_img
  
  def generate_single_data(self):
    img_w, img_h = self.get_size()
    img = self.get_blank(img_w, img_h)
    
    jng_dict, label = self.get_label_dict(img_aspect_ratio = img_h/img_w)
    
    jng_img = self.generate_image_by_dict(img, jng_dict)
    
    return label, img_w, img_h
  
  # label generator
  @classmethod
  def get_label_dict(cls, img_aspect_ratio = 1.0, div=None):
    pitch_range = cls.get_pitch_range()
    jng_dict = cls.get_jng_dict( pitch_range, div=( div if img_aspect_ratio >= 1.0 else randint(1, 2) ) )
    
    label = JeongganProcessor.get_label(jng_dict)
    
    return jng_dict, label
    
  @staticmethod
  def get_pitch_range(): # len 5 ~ 8
    num_pitch = len(PITCH_ORDER)
    
    center = randint(0, num_pitch)
    offset = bool(randint(0, 1)) # True: center is octave_width//2 / False: center is octave_width//2 + 1
    
    min_idx = 4 if offset else 5
    max_idx = 5 if offset else 4
    
    res = []
    
    if center < min_idx:
      res = PITCH_ORDER[:OCTAVE_WIDTH]
      
    elif center > num_pitch - max_idx:
      res = PITCH_ORDER[num_pitch-OCTAVE_WIDTH:]
    
    else:
      res = PITCH_ORDER[center-min_idx:center+max_idx]
    
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
        cur = choice(plist)
        
        cols.append(cur)
      
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
    
  def generate_image_by_dict(self, img, dict):
    img_h, img_w = img.shape[:2]
    
    row_div = dict['row_div']
    
    note_height = 40 if row_div == 1 else 36
    
    row_heights = []
    for row in dict['rows']:
      if row['col_div'] == 1 and row['cols'][0] == 'conti':
        row_heights.append(MARK_HEIGHT)
        continue
      
      row_heights.append(note_height)

    gap = (img_h - sum(row_heights)) // (row_div + 1)

    row_template = [ gap + idx * (row_heights[idx - 1 if idx > 0 else idx] + gap) for idx in range(row_div) ]

    for row_idx, row in enumerate(dict['rows']):
      col_div = row['col_div']
      
      notes = [] 
      for note_name in row['cols']:
        note_img_path = self.img_path_dict[note_name]
        
        if isinstance(note_img_path, list):
          note_img_path = choice(note_img_path)
        
        note_img = cv2.imread(note_img_path, cv2.IMREAD_UNCHANGED)
        
        if note_name in ('conti', 'pause'):
          note_img = self.make_mark(note_img, note_name)
        else:
          note_img = self.resize_img_by_height(note_img, note_height)
        
        note_img = self.remove_background(note_img)
        
        notes.append((note_img, note_name))
      
      gap = ( img_w - sum( [ note[0].shape[1] for note in notes ] ) ) // (col_div + 1)
      col_template = [ gap + idx * (notes[idx - 1 if idx > 0 else idx][0].shape[1] + gap) for idx in range(col_div) ] if col_div > 1 else [ round(img_w / 2) - round(notes[0][0].shape[1] / 2) ]
      
      for col_idx, note_tuple in enumerate(notes):
        note, note_name = note_tuple
        
        pos_x = col_template[col_idx]
        pos_y = row_template[row_idx]
        
        if note_name in ('conti', 'pause'):
          pos_y += (note_height - note.shape[0]) // 2
      
        img = self.insert_img(img, note, pos_x, pos_y)
    
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

    # normalize alpha to 0 ~ 1
    src_alpha = src[y:y+h, x:x+w, 3] / 255.0
    ins_alpha = insert[:, :, 3] / 255.0

    # blend src and insert channel by channel
    for ch in range(0, 3):
      src[y:y+h, x:x+w, ch] = ins_alpha * insert[:, :, ch] + \
                              src_alpha * src[y:y+h, x:x+w, ch] * (1 - ins_alpha)

    # denoramlize alpha to 0 ~ 255
    src[y:y+h, x:x+w, 3] = (1 - (1 - ins_alpha) * (1 - src_alpha)) * 255
    
    return src

  @staticmethod
  def resize_img_by_height(img, target_height):
    og_h, og_w = img.shape[:2]
    
    resize_ratio = target_height / og_h
    resize_width = round(og_w * resize_ratio)
    
    return cv2.resize(img, dsize=(resize_width, target_height), interpolation=None)
  
  @classmethod
  def make_mark(cls, img, name):
    width = MARK_WIDTHS[name]
    img_rs = cv2.resize(img, dsize=( width, round(img.shape[0]/img.shape[1] * width) ))
    
    bg = cls.get_blank(width, MARK_HEIGHT)
    
    return cls.insert_img(bg, img_rs, 0, MARK_HEIGHT//2 - img_rs.shape[0]//2)