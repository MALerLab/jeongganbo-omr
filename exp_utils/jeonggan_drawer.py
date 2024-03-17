import matplotlib.pyplot as plt
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
    self.main_rect_height = 10.5
    self.upper_margin_height = 1
    self.dpi = dpi
    note_img_path_dict = get_img_paths('test/synth/src', ['notes', 'symbols'])
    self.synth = JeongganSynthesizer(note_img_path_dict)
    self.maximum_gak_width = self.main_rect_width / 21
  

  def draw_blank_page(self,
                      num_gaks = 10,
                      num_jeonggans_per_gak = 20,
                      jg_width_ratio = 1.2):

    fig, ax = plt.subplots(figsize=(self.page_width, self.page_height), dpi=self.dpi)  # A4 paper size in inches
    # Set the limits of the plot
    ax.set_xlim(0, 8.27)
    ax.set_ylim(0, 11.69)

    rectangle_x = self.page_left_margin  # X-coordinate of the top-left corner of the rectangle
    rectangle_y = self.page_bottom_margin  # Y-coordinate of the top-left corner of the rectangle

    # Draw the rectangle
    gak_width = self.main_rect_width / (num_gaks * 2 + 1)
    if gak_width > self.maximum_gak_width:
      gak_width = self.maximum_gak_width
    main_rect_width = gak_width * (2 * num_gaks + 1)
    rectangle_x = (self.page_width - main_rect_width) - self.page_left_margin
      
    rectangle = plt.Rectangle((rectangle_x, rectangle_y), main_rect_width, self.main_rect_height, edgecolor='black', facecolor='none')
    ax.add_patch(rectangle)

    # Draw upper margin block
    upper_margin_y = self.page_bottom_margin + self.main_rect_height - self.upper_margin_height

    upper_margin_block = plt.Rectangle((rectangle_x, upper_margin_y), main_rect_width, self.upper_margin_height, edgecolor='black', facecolor='none')
    ax.add_patch(upper_margin_block)


    # Draw Gaks
    gak_positions = [rectangle_x + main_rect_width - gak_width * (2 * i + 2) for i in range(num_gaks)]
    for gak_x in gak_positions:
      # draw line from top margin_box to bottom of main rectangle
      ax.plot([gak_x, gak_x], [rectangle_y, rectangle_y + self.main_rect_height - self.upper_margin_height], color='black', linewidth=0.5)
      ax.plot([gak_x+gak_width * jg_width_ratio, gak_x+gak_width * jg_width_ratio], [rectangle_y, rectangle_y + self.main_rect_height - self.upper_margin_height], color='black', linewidth=0.5)

    # Draw Daegang line 
    jeonggan_height = (self.main_rect_height - self.upper_margin_height) / num_jeonggans_per_gak
    jeonggan_positions = [rectangle_y + jeonggan_height * i for i in range(num_jeonggans_per_gak)]

    for i, jeonggan_y in enumerate(reversed(jeonggan_positions)):
      for gak_x in gak_positions:
        ax.plot([gak_x, gak_x + gak_width*jg_width_ratio], [jeonggan_y, jeonggan_y], color='black', linewidth=0.5)
      if i in (5, 9, 13): # TODO: make this more general
        ax.plot([rectangle_x, rectangle_x + main_rect_width], [jeonggan_y, jeonggan_y], color='black', linewidth=1)
    
    jg_positions = [(gak_x, jeonggan_y) for gak_x in gak_positions for jeonggan_y in jeonggan_positions]
    jg_positions.sort(key=lambda x: (-x[0], -x[1] ))

    jg_width = gak_width * jg_width_ratio
    jg_height = jeonggan_height

    return fig, ax, jg_positions, (jg_width, jg_height)
  
  def draw_jeonggan(self, jeonggan_labels, ax, jg_positions, w_h_in_inches):
    assert len(jeonggan_labels) == len(jg_positions)
    w, h = w_h_in_inches
    for label, jg_position in zip(jeonggan_labels, jg_positions):
      if label == '0:5': continue
      jng_dict = self.synth.label2dict(label)
      img = self.synth.get_blank(width=int(w*self.dpi), height=int(h*self.dpi))
      jng_img = self.synth.generate_image_by_dict(img, jng_dict, apply_noise=False)
      ax.imshow(jng_img, extent=(jg_position[0], jg_position[0] + w, jg_position[1], jg_position[1] + h))


if __name__ == '__main__':
  import argparse
  from PIL import Image
  from pathlib import Path
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_path', type=str, required=True)
  parser.add_argument('--output_path', type=str, default='test/jg_page/')
  
  args = parser.parse_args()
  Path(args.output_path).mkdir(parents=True, exist_ok=True)
  
  with open(args.input_path, 'r') as f:
    jg_text = f.read()

  jg_by_part = jg_text.split('\n\n')
  jg_by_part = [jg.split('\n') for jg in jg_by_part]
  
  drawer = JeongganboPageDrawer()
  part_names = ['대금', '피리', '해금', '가야금', '거문고']
  image_paths = []
  
  for i, part in enumerate(jg_by_part):
    for j in range(ceil(len(part)/10)):
      selected_gaks = part[j*10:(j+1)*10]
      num_gaks = len(selected_gaks)
      jgs = [jg for gak in selected_gaks for jg in gak.split('|')]
      fig, ax, jg_positions, w_h = drawer.draw_blank_page(num_gaks=num_gaks)
      drawer.draw_jeonggan(jgs, ax, jg_positions, w_h)
      ax.axis('off')
      plt.savefig(f'{args.output_path}{part_names[i]}_{j}.png', bbox_inches='tight', pad_inches=0, dpi=300)
      plt.close()
      print(f'{part_names[i]}_{j} is saved')
      image_paths.append(f'{args.output_path}{part_names[i]}_{j}.png')
      
  first_image = Image.open(image_paths[0]).convert('RGB')
  other_images = [Image.open(image).convert('RGB') for image in image_paths[1:]]
  
  pdf_path = args.output_path + 'jeongganbo.pdf'
  first_image.save(pdf_path, save_all=True, append_images=other_images)

  

