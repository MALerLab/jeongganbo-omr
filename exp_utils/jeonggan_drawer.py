import matplotlib.pyplot as plt
import koreanize_matplotlib
from math import ceil
from .jeonggan_synthesizer import JeongganSynthesizer, get_img_paths

class JeongganboPageDrawer:
  def __init__(self,
               page_width=8.27,
               page_height=11.69,
               page_left_margin=0.6,
               page_bottom_margin=0.5,
               dpi = 300):
    self.page_width = page_width
    self.page_height = page_height
    self.page_left_margin = page_left_margin
    self.page_bottom_margin = page_bottom_margin
    self.main_rect_width = 7
    self.main_rect_height = 10.7
    self.upper_margin_height = 1
    self.dpi = dpi
    note_img_path_dict = get_img_paths('test/synth/src', ['notes', 'symbols'])
    self.synth = JeongganSynthesizer(note_img_path_dict)
    self.maximum_gak_width = self.main_rect_width / 21

  def draw_blank_page(self,
                      num_cols = 10,
                      num_jeonggans_per_gak = 10,
                      jg_width_ratio = 1.2,
                      daegang_length = (3,2,2,3),
                      is_title_page = False):

    fig, ax = plt.subplots(figsize=(self.page_width, self.page_height), dpi=self.dpi)
    ax.set_xlim(0, self.page_width)
    ax.set_ylim(0, self.page_height)

    rectangle_x = self.page_left_margin
    rectangle_y = self.page_bottom_margin
    gak_width = min(self.main_rect_width / (num_cols * 2 + 1), self.maximum_gak_width)
    gak_height = (0.5 * self.main_rect_height) - self.upper_margin_height
    jg_width = gak_width * jg_width_ratio
    jg_height = gak_height / num_jeonggans_per_gak

    # Draw main border rectangle
    rectangle = plt.Rectangle((rectangle_x, rectangle_y),
                               self.main_rect_width,
                               self.main_rect_height,
                               edgecolor='black',
                               facecolor='none')
    ax.add_patch(rectangle)

    if is_title_page:
      title_border_x = rectangle_x + self.main_rect_width - (gak_width * 2)
      ax.plot([title_border_x, title_border_x],
              [rectangle_y, rectangle_y + self.main_rect_height],
              color='black',
              linewidth=1)
      margin_block_width = title_border_x - rectangle_x
    else:
      margin_block_width = self.main_rect_width

    # Draw margin blocks
    upper_margin_y = self.page_bottom_margin + self.main_rect_height - self.upper_margin_height
    upper_margin_block = plt.Rectangle((rectangle_x, upper_margin_y),
                                       margin_block_width,
                                       self.upper_margin_height,
                                       edgecolor='black',
                                       facecolor='none')
    mid_margin_block = plt.Rectangle((rectangle_x,
                                      rectangle_y + gak_height),
                                      margin_block_width,
                                      self.upper_margin_height,
                                      edgecolor='black',
                                      facecolor='none')
    ax.add_patch(upper_margin_block)
    ax.add_patch(mid_margin_block)

    r = range(1, num_cols) if is_title_page else range(num_cols)
    gak_x_positions = [rectangle_x + self.main_rect_width - gak_width * (2 * i + 2) for i in r]

    # Draw Gaks
    for gak_x in gak_x_positions:
      left_x = gak_x
      right_x = gak_x + jg_width

      ax.plot([left_x, left_x],
              [rectangle_y, rectangle_y + gak_height],
              color='black',
              linewidth=0.5)
      ax.plot([left_x, left_x],
              [rectangle_y + (0.5 * self.main_rect_height),
               rectangle_y + self.main_rect_height - self.upper_margin_height],
              color='black',
              linewidth=0.5)
      
      ax.plot([right_x, right_x],
              [rectangle_y, rectangle_y + gak_height],
              color='black',
              linewidth=0.5)
      ax.plot([right_x, right_x],
              [rectangle_y + (0.5 * self.main_rect_height),
               rectangle_y + self.main_rect_height - self.upper_margin_height],
              color='black',
              linewidth=0.5)

    # Draw Jeonggan cells
    daegang_boundaries = [-1]
    for i, length in enumerate(daegang_length[:-1]):
      daegang_boundaries.append(daegang_boundaries[-1] + length)
    print(daegang_boundaries)

    jg_y_upper = [rectangle_y + (0.5 * self.main_rect_height) + jg_height * i for i in range(num_jeonggans_per_gak)]
    jg_y_lower = [rectangle_y + jg_height * i for i in range(num_jeonggans_per_gak)]

    for i, jeonggan_y in enumerate(reversed(jg_y_upper)):
      for gak_x in gak_x_positions:
        ax.plot([gak_x, gak_x + gak_width*jg_width_ratio], [jeonggan_y, jeonggan_y], color='black', linewidth=0.5)
      if i in daegang_boundaries[1:]:
        if is_title_page:
          x2 = title_border_x
        else:
          x2 = rectangle_x + self.main_rect_width
        ax.plot([rectangle_x, x2], [jeonggan_y, jeonggan_y], color='black', linewidth=1)
    
    for i, jeonggan_y in enumerate(reversed(jg_y_lower)):
      for gak_x in gak_x_positions:
        ax.plot([gak_x, gak_x + gak_width*jg_width_ratio], [jeonggan_y, jeonggan_y], color='black', linewidth=0.5)
      if i in daegang_boundaries[1:]:
        if is_title_page:
          x2 = title_border_x
        else:
          x2 = rectangle_x + self.main_rect_width
        ax.plot([rectangle_x, x2], [jeonggan_y, jeonggan_y], color='black', linewidth=1)
    
    jg_positions_upper = [(gak_x, jeonggan_y) for gak_x in gak_x_positions for jeonggan_y in jg_y_upper]
    jg_positions_lower = [(gak_x, jeonggan_y) for gak_x in gak_x_positions for jeonggan_y in jg_y_lower]
    jg_positions_upper.sort(key=lambda x: (-x[0], -x[1]))
    jg_positions_lower.sort(key=lambda x: (-x[0], -x[1]))
    jg_positions = jg_positions_upper + jg_positions_lower

    return fig, ax, jg_positions, (jg_width, jg_height)
  
  def draw_jeonggan(self, jeonggan_labels, ax, jg_positions, w_h_in_inches):
    print(f"Jeonggan cells populated: {len(jeonggan_labels)} / {len(jg_positions)}")
    assert len(jeonggan_labels) <= len(jg_positions)
    w, h = w_h_in_inches
    for label, jg_position in zip(jeonggan_labels, jg_positions, strict=False):
      if label == '-:5': continue
      print(label)
      jng_dict = self.synth.label2dict(label)
      img = self.synth.get_blank(width=int(w*self.dpi), height=int(h*self.dpi))
      jng_img = self.synth.generate_image_by_dict(img, jng_dict, apply_noise=False, random_symbols=False)
      ax.imshow(jng_img, extent=(jg_position[0], jg_position[0] + w, jg_position[1], jg_position[1] + h))