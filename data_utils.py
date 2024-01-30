import cv2
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from pathlib import Path
from omr_cnn import Inferencer
import torch


JG_MIN_AREA = 8000
JG_MAX_AREA = 20000
JG_MIN_WIDTH = 85
JG_MAX_WIDTH = 110
JG_MIN_HEIGHT = 66
JG_MAX_HEIGHT = 160

MIN_X_GAP = 5
MIN_Y_GAP = 5
GAK_BREAK_GAP = 40

TITLE_MIN_WIDTH = 120
TITLE_MAX_WIDTH = 450
TITLE_MIN_HEIGHT = 800

JANGDAN_GAK_POS_MIN_DIFF = 4
PAGE_HEIGHT = 3091



class JeongganboReader:
  def __init__(self, run_omr=False) -> None:
    
    self.line_min_length = 70
    self.line_min_thickness = 2
    self.thick_line_thickness = 5

    self.kernel_h = np.ones((self.line_min_thickness, self.line_min_length), np.uint8)
    self.kernel_v = np.ones((self.line_min_length, self.line_min_thickness), np.uint8)
    self.thick_kernel_h = np.ones((self.thick_line_thickness, self.line_min_length), np.uint8)
    self.thick_kernel_h_alt = np.ones((self.line_min_thickness, self.line_min_length * 3), np.uint8)
    self.thick_kernel_v = np.ones((self.line_min_length, self.thick_line_thickness), np.uint8)
    self.final_kernel = np.ones((2, 2), np.uint8)

    self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))

    self.run_omr = run_omr
    if self.run_omr:
      self.omr = Inferencer()

  def _repair_v_lines(self, img_bin_v: np.ndarray, img_bin_h:np.ndarray) -> np.ndarray:
    # img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, self.kernel_v)

    contours, _ = cv2.findContours(img_bin_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_contours, _ = cv2.findContours(img_bin_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0 or len(h_contours) == 0:
      return img_bin_v, h_contours
    v_contours = np.asarray([cv2.boundingRect(c) for c in contours])
    h_contours = np.asarray([cv2.boundingRect(c) for c in h_contours])
    v_contour_counter = Counter(v_contours[:,1])

    # repair from top
    most_common_y_pos = v_contour_counter.most_common(1)[0][0]
    filtered_contours = v_contours[v_contours[:,1] == most_common_y_pos]
    most_common_len = Counter(filtered_contours[:,3]).most_common(1)[0][0]
    broken_contours = filtered_contours[filtered_contours[:,3] < most_common_len]
    for contour in broken_contours:
        x, y, w, h = contour
        # check whether broken end is by h_contour
        if len(h_contours[h_contours[:,1]+h_contours[:,3] == y+h]) > 0:
          continue
        img_bin_v[y:y+most_common_len, x:x+w] = 255
    
    # repair from bottom
    most_common_end_y = Counter(v_contours[:,1] + v_contours[:,3]).most_common(1)[0][0]
    filtered_contours = v_contours[ abs(v_contours[:,1] + v_contours[:,3]  - most_common_end_y) < 5 ]

    broken_contours = filtered_contours[filtered_contours[:,3] < most_common_len]
    for contour in broken_contours:
        x, y, w, h = contour
        # check whether broken end is by h_contour
        if len(h_contours[h_contours[:,1] == y]) > 0:
          continue
        img_bin_v[y+h-most_common_len:y+h, x:x+w] = 255
    
    too_short_contours = v_contours[v_contours[:,3] < JG_MAX_HEIGHT]
    for contour in too_short_contours:
        x, y, w, h = contour
        img_bin_v[y:y+h, x:x+w] = 0
    return img_bin_v, h_contours
  

  def _get_boxes(self, img_bin: np.ndarray) -> np.ndarray:
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, self.kernel_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, self.kernel_v)
    img_bin_v, h_contours = self._repair_v_lines(img_bin_v, img_bin_h)

    img_bin_final = img_bin_h + img_bin_v
    img_bin_final = cv2.dilate(img_bin_final, self.final_kernel, iterations=1)
    img_bin_final = cv2.morphologyEx(img_bin_final, cv2.MORPH_CLOSE, self.closing_kernel)

    _, _, boxes, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=4, ltype=cv2.CV_32S)

    # is_large_enough = boxes[:,-1] > JG_MIN_AREA
    # is_small_enough = boxes[:,-1] < JG_MAX_AREA
    # boxes = boxes[is_large_enough & is_small_enough]

    is_wide_enough = boxes[:,2] > JG_MIN_WIDTH
    is_narrow_enough = boxes[:,2] < JG_MAX_WIDTH
    is_tall_enough = boxes[:,3] > JG_MIN_HEIGHT
    is_short_enough = boxes[:,3] < JG_MAX_HEIGHT * 2

    boxes = boxes[is_wide_enough & is_narrow_enough & is_tall_enough & is_short_enough]

    return boxes, h_contours
  
  def _get_thick_lines(self, img_bin: np.ndarray) -> np.ndarray:
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, self.thick_kernel_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, self.thick_kernel_v)
    img_bin_final = img_bin_h + img_bin_v 

    img_bin_h_alt = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, self.thick_kernel_h_alt)
    img_bin_final = img_bin_h_alt + img_bin_v
    # img_bin_final += img_bin_h_alt

    img_bin_final = cv2.dilate(img_bin_final, self.final_kernel, iterations=1)

    _, _, boxes, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=4, ltype=cv2.CV_32S)
    boxes = boxes[boxes[:,-1] > JG_MIN_AREA]

    contours, _ = cv2.findContours(img_bin_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_contours = [cv2.boundingRect(contour) for contour in contours]
    h_contours = np.asarray(h_contours)

    return boxes, h_contours

  def _detect_title_box(self, img_bin, jeonggan_boxes: np.ndarray, thick_boxes: np.ndarray, margin:int=8) -> np.ndarray:
    # boxes = result of cv2.connectedComponentsWithStats
    # boxes = [x, y, w, h, area]

    # filter by width
    is_wide_enough = thick_boxes[:,2] > TITLE_MIN_WIDTH
    is_narrow_enough = thick_boxes[:,2] < TITLE_MAX_WIDTH
    # filter by height
    is_tall_enough = thick_boxes[:,3] > TITLE_MIN_HEIGHT

    # return thick_boxes[is_wide_enough & is_narrow_enough]
    boxes = thick_boxes[is_wide_enough & is_narrow_enough & is_tall_enough]

    # for candidate in candidates, check whether it is total blank
    # if it is total blank, remove it from candidates
    # if it is not total blank, return it
    jeonggan_xs = np.unique(jeonggan_boxes[:,0])
    left_most_jeonggan_x = jeonggan_xs.min()

    not_blank_boxes = []
    for box in boxes:
      x, y, w, h = box[:-1]
      if np.sum(img_bin[y+margin:y+h-margin, x+margin:x+w-margin]) != 0 \
        and ((jeonggan_xs - x) * (jeonggan_xs - (x+w)) > 0).all() \
        and x > left_most_jeonggan_x:
      # if len(np.nonzero(img_bin[y:y+h, x:x+w])[0]) > w * 1.5:
        not_blank_boxes.append(box)
    return np.asarray(not_blank_boxes)


  def _process_img(self, image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    img_bin = 255 - img_bin
    return img_bin

  def _detect_double_h_line(self, img_bin_h: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(img_bin_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_contours = [cv2.boundingRect(contour) for contour in contours]
    h_contours = np.asarray(h_contours)

    return 
  # def detect_jangdan_gak_from_img_path(self, image_path: str) -> np.ndarray:
  #   image = cv2.imread(image_path)
  #   img_bin = self._process_img(image)

  #   small_boxes = self._get_boxes(img_bin)
  #   boxes, h_contours, = self._get_thick_lines(img_bin)

  #   boxes = self._detect_jangdan_gak(small_boxes, boxes)


  def detect_title_box_from_img_path(self, image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)

    img_bin = self._process_img(image)

    jeonggan_boxes = self._get_boxes(img_bin)
    if len(jeonggan_boxes) == 0:
      return None
    thick_boxes, h_contours = self._get_thick_lines(img_bin)
    title_box = self._detect_title_box(img_bin, jeonggan_boxes, thick_boxes)

    if len(title_box) == 0:
      return None
    print(f"{len(title_box)} Title box detected at {image_path}")
    box = title_box[0]
    return image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

  def _split_page_by_title(self, image, jeonggan_boxes, thick_boxes, h_contours, title_box):
    split_x_pos = title_box[0] + title_box[2]
    right_page = image[:, split_x_pos:]
    left_page = image[:, :split_x_pos]

    right_jeonggan_boxes = jeonggan_boxes[jeonggan_boxes[:,0] >= split_x_pos]
    right_jeonggan_boxes[:,0] -= split_x_pos
    left_jeonggan_boxes = jeonggan_boxes[jeonggan_boxes[:,0] < split_x_pos]

    right_thick_boxes = thick_boxes[thick_boxes[:,0] >= split_x_pos]
    right_thick_boxes[:,0] -= split_x_pos
    left_thick_boxes = thick_boxes[thick_boxes[:,0] < split_x_pos]

    right_h_contours = h_contours[h_contours[:,0] >= split_x_pos]
    right_h_contours[:,0] -= split_x_pos
    left_h_contours = h_contours[h_contours[:,0] < split_x_pos]

    right_page = Page(right_page, right_jeonggan_boxes, right_thick_boxes, right_h_contours, None)
    left_page = Page(left_page, left_jeonggan_boxes, left_thick_boxes, left_h_contours, title_box)

    # return right_page, left_page
    return left_page, right_page

  def adjust_box_position(self, boxes, margin=4):
    if boxes is None or len(boxes) == 0:
      return boxes
    boxes[:,0] -= margin
    boxes[:,1] -= margin
    boxes[:,2] += margin
    boxes[:,3] += margin
    return boxes

  def filter_blank_jeonggan_after_double_line(self, jeonggan_boxes, h_contours, image, margin=2):
    h_contours = np.asarray(sorted(h_contours.tolist(), key=lambda x: (x[0], x[1]) ))
    diff = np.diff(h_contours, axis=0, prepend=[[0,0,0,0]])
    condition = (diff[:,0] == 0) * (diff[:,1] < 12)
    filtered_contours = h_contours[condition]
    if len(filtered_contours) == 0:
      return jeonggan_boxes
    for box in filtered_contours:
      box_index = np.nonzero(( abs(jeonggan_boxes[:,:2] - box[:2]).sum(axis=1) < 3))[0]
      if len(box_index) == 0:
        continue
      box_index = box_index[0]
      jeonggan_box = jeonggan_boxes[box_index]
      jeonggan_box_img = image[jeonggan_box[1]:jeonggan_box[1]+jeonggan_box[3], jeonggan_box[0]:jeonggan_box[0]+jeonggan_box[2]]
      if (jeonggan_box_img[margin:-margin, margin:-margin] < 80 ).sum() == 0:
        jeonggan_boxes = np.delete(jeonggan_boxes, box_index, axis=0)
    return jeonggan_boxes

  def run_omr_on_page(self, page):
    # page: Page object
    for jeonggan in page.jeonggan_list:
      jeonggan_img = jeonggan.img
      jeonggan.omr_text = self.omr(jeonggan_img)

  def __call__(self, image, return_title_detected=False):
    if isinstance(image, str):
      image = cv2.imread(image)
    elif isinstance(image, Path):
      image = cv2.imread(str(image))
    img_bin = self._process_img(image)

    jeonggan_boxes, thin_h_contours = self._get_boxes(img_bin)
    if len(jeonggan_boxes) == 0:
      return Page(image, [], [], []), False
    thick_boxes, h_contours = self._get_thick_lines(img_bin)
    title_box = self._detect_title_box(img_bin, jeonggan_boxes, thick_boxes)
    # assert len(title_box) <= 1, f"More than 1 title box detected: {len(title_box)}"
    if len(title_box) > 0:
      title_box_detected = True
      title_box = title_box[0]
      right_most_jeonggan_x = jeonggan_boxes[:,0].max()
      if title_box[0] + 3 < right_most_jeonggan_x: # 3 is margin
        print(f"Title box is detected but it is not on the right side of the page")
        # Page has to be splitted
        return self._split_page_by_title(image, jeonggan_boxes, thick_boxes, h_contours, title_box)
    else:
      title_box_detected = False
      title_box = None
    jeonggan_boxes = self.adjust_box_position(jeonggan_boxes)
    thick_boxes = self.adjust_box_position(thick_boxes)

    jeonggan_boxes = self.filter_blank_jeonggan_after_double_line(jeonggan_boxes, thin_h_contours, image)
    page = Page(image, jeonggan_boxes, thick_boxes, h_contours, title_box)
    if self.run_omr:
      self.run_omr_on_page(page)
    if return_title_detected:
      return page, title_box_detected
    return page
  
  def parse_multiple_pages(self, image_paths:List[str], min_jeonggan_count=4):
    pieces = []
    temp_pages = []
    for image_path in image_paths:
      print(f"Processing {image_path}")
      if isinstance(image_path, Path):
        image_path = str(image_path)
      page, is_title_included = self(image_path, return_title_detected=True)
      if len(page) == 0:
        print(f"No jeonggan detected at {image_path}")
        if len(temp_pages) > 0:
          pieces.append(Piece(temp_pages))
          temp_pages = []
        continue
      if len(page) < min_jeonggan_count:
        print(f"Too few jeonggan detected at {image_path}, {len(page)}")
        if len(temp_pages) > 0:
          pieces.append(Piece(temp_pages))
          temp_pages = []
        continue
      if isinstance(is_title_included, bool) and is_title_included and len(temp_pages) > 0: # new piece start
        print(f"New piece detected at {image_path}")
        pieces.append(Piece(temp_pages))
        temp_pages = []
      elif isinstance(is_title_included, Page): # two pages are returned, which mean title detected
        print(f"New piece detected at {image_path}")
        temp_pages.append(is_title_included)
        pieces.append(Piece(temp_pages))
        temp_pages = []
      temp_pages.append(page)
    if len(temp_pages) > 0:
      pieces.append(Piece(temp_pages))
    return pieces
  


class NoteSymbol:
  def __init__(self) -> None:
    self.pitch = None
    self.beat_in_jeonggan = None

  def __repr__(self) -> str:
    return f"NoteSymbol at {self.beat_in_jeonggan}, pitch: {self.pitch}"

class Jeonggan:
  def __init__(self, img, x, y, w, h) -> None:
    self.img = img[y:y+h, x:x+w]
    self.x = x
    self.y = y
    self.w = w
    self.h = h

    self.org_x = x
    self.org_y = y
    self.is_jangdan = False
    self.gak_id = 0
    self.daegang_id = 0
    self.beat = 0
    self.piece_beat = None

    self.is_double = self.h > JG_MAX_HEIGHT
    self.notes = None
    self.omr_text = None

    # assert self.w < JG_MAX_WIDTH and self.w > JG_MIN_WIDTH, f"Jeonggan width is not in range: {self.w}"
    # assert self.h < JG_MAX_HEIGHT and self.h > JG_MIN_HEIGHT, f"Jeonggan height is not in range: {self.h}"

  def _parse_notes(self) -> List[NoteSymbol]:

    return

  def __repr__(self) -> str:
    return f"Jeonggan at Gak {self.gak_id}, Daegang {self.daegang_id}, Beat {self.beat}, Piece Beat {self.piece_beat}, img ({self.x}, {self.y})"
  


class Gak:
  def __init__(self, jeonggans: List[Jeonggan], row_id: int) -> None:
    self.jeonggans = jeonggans
    self.x = jeonggans[0].x
    self.y = jeonggans[0].y
    self.w = jeonggans[0].w
    self.h = jeonggans[-1].y + jeonggans[-1].h - jeonggans[0].y

    self.org_x = jeonggans[0].org_x
    self.org_y = jeonggans[0].org_y
    self.row_id = row_id
    self.order_id = None
    self.piece_order_id = None
    self.is_jangdan = False
    self.is_independent = False # For 돌장
    self.start_beat = 0
    self.daegang_start_beats = [0]

    self.mean_h = self.h / len(self)
    assert self.mean_h < JG_MAX_HEIGHT and self.mean_h > JG_MIN_HEIGHT, f"Jeonggan height is not in range: {self.mean_h}, {[x.h for x in self.jeonggans]}"

  def __len__(self) -> int:
    return len(self.jeonggans) + len([jg for jg in self.jeonggans if jg.is_double])
  
  def __repr__(self) -> str:
    if self.is_jangdan:
      return f"Jangdan Gak at Row {self.row_id} ({self.x}, {self.y}) with {len(self.jeonggans)} jeonggans. Avg height: {self.mean_h:.2f}"
    return f"Gak at Row {self.row_id} ({self.x}, {self.y}) with {len(self.jeonggans)} jeonggans. Avg height: {self.mean_h:.2f}, Start Beat: {self.start_beat}"
  


class Page:
  def __init__(self, img:np.ndarray, boxes, thick_boxes, thick_h_contours, title_box=None) -> None:
    self.img = img
    self.boxes = boxes
    self.thick_boxes = thick_boxes
    self.thick_h_contours = thick_h_contours
    self.title_box = title_box

    self.jeonggan_boxes = boxes
    if len(self.jeonggan_boxes) == 0: # no jeonggan detected
      self.jeonggan_list = []
      self.gaks = []
      self.new_gak_start_y_pos = []
      return
    self.gak_x_positions, self.x_map_dict, self.jeonggan_y_positions, self.y_map_dict = self._get_gak_xy_positions(self.jeonggan_boxes)

    self.jeonggan_list = [Jeonggan(img, x, y, w, h) for x, y, w, h in self.jeonggan_boxes[:,:-1]]
    self._update_jeonggan_position(self.jeonggan_list)
    self.jeonggan_list = self._sort_jeonggan_by_position(self.jeonggan_list)
    self.gaks, self.new_gak_start_y_pos = self._detect_gak(self.jeonggan_list)
    self._update_gak_beats()
    self._update_jeonggan_beat()
    self._update_daegang()
    self._update_jeonggan_daegang()

    self._detect_jangdan_gak()

    self.jeonggan_list = [jeonggan for gak in self.gaks for jeonggan in gak.jeonggans]

  def _get_gak_xy_positions(self, jeonggan_boxes) -> Tuple[list, dict]:
    x_values = np.unique(jeonggan_boxes[:,0])
    x_cleaned_list, x_map_dict = self._get_unique_position(x_values, MIN_X_GAP)

    y_values = np.unique(jeonggan_boxes[:,1])
    y_cleaned_list, y_map_dict = self._get_unique_position(y_values, MIN_Y_GAP)

    return x_cleaned_list, x_map_dict, y_cleaned_list, y_map_dict
  
  def _get_unique_position(self, values:np.ndarray, min_gap:int):
    cleaned_list = [values[0]]
    ommited_list = {}
    for i in range(1, len(values)):
      if values[i] - cleaned_list[-1] > min_gap:
        cleaned_list.append(values[i])
      else:
        ommited_list[values[i]] = cleaned_list[-1]
    cleaned_list = sorted(list(set(cleaned_list)))
    mapping_dict = {cleaned_list[i]: cleaned_list[i] for i in range(len(cleaned_list))}
    mapping_dict.update(ommited_list)
    return cleaned_list, mapping_dict

  
  def _detect_gak(self, jeonggan_list) -> int:
    row_break_y_pos = []
    gaks = []
    temp_gak = [jeonggan_list[0]]
    for i in range(1, len(jeonggan_list)):
      cur = jeonggan_list[i]
      prev = temp_gak[-1]
      if cur.x == prev.x:
        if cur.y - (prev.y + prev.h) > GAK_BREAK_GAP:
          row_break_y_pos.append(cur.y)
          gaks.append(temp_gak)
          temp_gak = [cur]
        else:
          temp_gak.append(cur)
      else:
        gaks.append(temp_gak)
        temp_gak = [cur]
    if len(temp_gak) > 0:
      gaks.append(temp_gak)
    row_break_y_pos = sorted(list(set(row_break_y_pos)))
    gaks = [Gak(gak, self._get_gak_row_id(gak, row_break_y_pos)) for gak in gaks]
    gaks = self._sort_gak(gaks)

    return gaks, [min([gak.y for gak in gaks])] + row_break_y_pos
  
  def _update_gak_beats(self) -> None:
    row_ids = defaultdict(list)
    for gak in self.gaks:
      row_ids[gak.row_id].append(gak)
    rows = list(row_ids.values())

    uneven_rows = [row for row in rows if len(set([len(gak) for gak in row])) != 1]
    if len(uneven_rows) == 0:
      return
    
    for row in uneven_rows:
      gak_by_len = defaultdict(list)
      for gak in row:
        gak_by_len[len(gak)].append(gak)
      max_len = max(gak_by_len.keys())

      for gak_len in gak_by_len.keys():
        if gak_len == max_len:
          continue
        for gak in gak_by_len[gak_len]:
          # assert gak.y > gak_by_len[max_len][0].y, "Shorter Gak is not on bottom of the row"
          if gak.y > gak_by_len[max_len][0].y:
            gak.start_beat = max_len - len(gak)
          else:
            print(f"Warning: Shorter Gak is not on bottom of the row: {gak}")
            gak.is_independent = True
      # assert len(set([len(gak) for gak in row[1:]])) == 1, "Even without First Gak, beat length is not even"
      # assert row[0].y > row[1].y, "First Gak is not on bottom of the row"
      # row[0].start_beat = len(row[1]) - len(row[0])

  def _update_jeonggan_beat(self):
    for gak in self.gaks:
      for i, jeonggan in enumerate(gak.jeonggans):
        jeonggan.beat = gak.start_beat + i

  def _detect_jangdan_gak(self):
    thick_boxes = self._concat_continuous_thick_boxes_in_y_pos(self.thick_boxes)
    thick_boxes = thick_boxes[:, :3]
    for gak in self.gaks:
      gak_pos = np.asarray([gak.org_x, gak.org_y, gak.w])
      min_diff = np.abs(thick_boxes - gak_pos).sum(axis=1).min()
      # gak_pos = np.asarray([gak.org_x, gak.org_y, gak.w, gak.h])
      # min_diff = np.abs(thick_boxes[:,:-1] - gak_pos).sum(axis=1).min()
      if min_diff < JANGDAN_GAK_POS_MIN_DIFF:
        gak.is_jangdan = True
        for jeonggan in gak.jeonggans:
          jeonggan.is_jangdan = True

  def _concat_continuous_thick_boxes_in_y_pos(self, thick_boxes):
    thick_box_in_list = thick_boxes.tolist()
    thick_box_in_list.sort(key=lambda x: (x[0], x[1]))

    concated = []
    for i in range(len(thick_box_in_list)-1):
      cur = thick_box_in_list[i]
      next = thick_box_in_list[i+1]
      if cur[0] == next[0] and abs(cur[1] + cur[3] - next[1]) < 8: # 8 is margin
        cur[3] += next[3] + (next[1] - cur[1] - cur[3])
        thick_box_in_list[i+1] = cur
      else:
        concated.append(cur)
    concated.append(thick_box_in_list[-1])
    return np.asarray(concated)
  
  
  def _update_jeonggan_position(self, jeonggan_list):
    for jeonggan in jeonggan_list:
      jeonggan.x = self.x_map_dict[jeonggan.x]
      jeonggan.y = self.y_map_dict[jeonggan.y]

  def _update_daegang(self):
    daegang_splitter_y_pos = []
    thick_y_pos = self.thick_boxes[:,1]
    daegang_pos_in_beat = []
    # find nearest position in self.jeonggan_y_positions
    for y in thick_y_pos:
      idx = np.abs(self.jeonggan_y_positions - y).argmin()
      if abs(self.jeonggan_y_positions[idx] - y) > 3: # not start of jeonggan
        continue
      # assert abs(self.jeonggan_y_positions[idx] - y) < 3, f"y position of daegang is not matched: {y}, {self.jeonggan_y_positions[idx]}"
      daegang_splitter_y_pos.append(self.jeonggan_y_positions[idx])
      daegang_pos_in_beat.append(idx)
    daegang_splitter_y_pos = sorted(list(set(daegang_splitter_y_pos)))
    daegang_pos_in_beat = sorted(list(set(daegang_pos_in_beat)))

    daegang_y_pos_by_row = [[] for _ in range(len(self.new_gak_start_y_pos)) ]
    row_id = 0

    row_y_positions = self.new_gak_start_y_pos + [PAGE_HEIGHT]

    for y_pos in daegang_splitter_y_pos:
      next_row_y_pos = row_y_positions[row_id + 1]
      if y_pos >= next_row_y_pos:
        row_id += 1
      daegang_y_pos_by_row[row_id].append(y_pos)
    
    for gak in self.gaks:
      daegang_y_pos = daegang_y_pos_by_row[gak.row_id]
      for y_pos in daegang_y_pos:
        gak.daegang_start_beats.append(self.jeonggan_y_positions.index(y_pos) - self.jeonggan_y_positions.index(self.new_gak_start_y_pos[gak.row_id]))
      gak.daegang_start_beats = sorted(list(set(gak.daegang_start_beats)))  

  def _update_jeonggan_daegang(self):
    for gak in self.gaks:
      for i in range(len(gak.daegang_start_beats)):
        cur_daegang_start_beat = gak.daegang_start_beats[i]
        next_daegang_start_beat = gak.daegang_start_beats[i+1] if i+1 < len(gak.daegang_start_beats) else len(gak)
        for jeonggan in gak.jeonggans[cur_daegang_start_beat:next_daegang_start_beat]:
          jeonggan.daegang_id = i

  def _sort_jeonggan_by_position(self, jeonggan_list):
    jeonggan_list.sort(key=lambda x: (-x.x, x.y))   
    return jeonggan_list
  
  def _sort_gak(self, gaks:List[Gak]):
    gaks = sorted(gaks, key=lambda x: (x.row_id, -x.x))
    for i in range(len(gaks)):
      gaks[i].order_id = i # update order_id
    return gaks
  
  def update_gak_info_to_jeonggan(self, gak_info):
    for jeonggan in self.jeonggan_list:
      jeonggan['gak'] = gak_info[jeonggan['x']]

  def _get_gak_row_id(self, jeonggan_list:List[Jeonggan], column_break_y_pos:List[int]):
    if jeonggan_list[0].y in column_break_y_pos:
      return column_break_y_pos.index(jeonggan_list[0].y) + 1
    return 0
  
  def __repr__(self) -> str:
    return f"Page with {len(self.jeonggan_list)} jeonggan"
  
  def __len__(self) -> int:
    return len([x for x in self.jeonggan_list if not x.is_jangdan])


class Piece:
  def __init__(self, pages:List[Page]) -> None:
    self.pages = pages
    self._update_gak_order_id()

  def __repr__(self) -> str:
    return f"Piece with {len(self.pages)} pages and {len(self.jeonggans)} jeonggans"

  def _update_gak_order_id(self):
    global_offset = 0
    for i in range(len(self.gaks)):
      self.gaks[i].piece_order_id = i
      for j, jeonggan in enumerate(self.gaks[i].jeonggans):
        jeonggan.gak_id = i
        jeonggan.piece_beat = global_offset + j
      global_offset += len(self.gaks[i])

  @property
  def gaks(self) -> List[Gak]:
    return [gak for page in self.pages for gak in page.gaks]

  @property
  def jeonggans(self) -> List[Jeonggan]:
    return [jeonggan for page in self.pages for jeonggan in page.jeonggan_list if not jeonggan.is_jangdan]


if __name__ == "__main__":
  import matplotlib.pyplot as plt

  image_path = 'haegeum_example.png'
  image_path = 'pngs/haegeum_pg-031.png'
  image_path = 'pngs/daegeum_pg-059.png'
  # image_path = 'haegeum_pg-150.png'
  # image_path = 'haegeum_pg-107.png'
  # image_path = 'haegeum_pg-236.png'
  # image_path = 'pngs/piri_pg-286.png'
  # image_path = 'pngs/piri_pg-026.png'
  image_path = 'pngs/gayageum_pg-271.png'

  reader = JeongganboReader()
  page = reader(image_path)
  print(page)
  print(page.gaks)
