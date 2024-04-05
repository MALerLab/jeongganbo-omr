import argparse
from pathlib import Path
from math import ceil
import matplotlib.pyplot as plt
from PIL import Image
from exp_utils import JeongganboPageDrawer

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_path', type=str, required=True)
  parser.add_argument('--output_path', type=str, default='test/jg_page/')
  parser.add_argument('--num_jeonggans_per_gak', type=int, default=10)
  parser.add_argument('--daegang_length', type=int, nargs='+', default=[3,2,2,3])
  parser.add_argument('--draw_titles', type=bool, default=True)
  
  args = parser.parse_args()
  print(args.daegang_length)
  Path(args.output_path).mkdir(parents=True, exist_ok=True)
  
  with open(args.input_path, 'r') as f:
    jg_text = f.read()

  jg_by_part = jg_text.split('\n\n')
  jg_by_part = [jg.split('\n') for jg in jg_by_part]
  for i, part in enumerate(jg_by_part):
    jg_by_part[i] = [gak for gak in part if len(gak.split('|')) > 1]
  
  drawer = JeongganboPageDrawer()
  # part_names = ['대금', '피리', '해금', '가야금', '거문고']
  # part_names = ['대금', '해금', '피리', '가야금', '거문고']
  part_names = ['대금', '피리', '해금', '아쟁', '가야금', '거문고']
  image_paths = []
  
  for i, part in enumerate(jg_by_part):
    for j in range(ceil(len(part)/20)):
      if args.draw_titles:
        selected_gaks = part[:18] if j == 0 else part[(j*20)-2:(j+1)*20-2]
      else:
        selected_gaks = part[j*20:(j+1)*20]
      num_cols = 10
      jgs = [jg for gak in selected_gaks for jg in gak.split('|')]
      fig, ax, jg_positions, w_h = drawer.draw_blank_page(num_cols=num_cols, 
                                                          num_jeonggans_per_gak=args.num_jeonggans_per_gak,
                                                          daegang_length=args.daegang_length,
                                                          is_title_page=args.draw_titles and j == 0)
      drawer.draw_jeonggan(jgs, ax, jg_positions, w_h)
      ax.axis('off')
      plt.text(0.5, 0.02, f"{part_names[i]} - {str(j+1)}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
      plt.savefig(f'{args.output_path}{part_names[i]}_{j}.png', bbox_inches='tight', pad_inches=0, dpi=300)
      plt.close()
      print(f'{part_names[i]}_{j} is saved')
      image_paths.append(f'{args.output_path}{part_names[i]}_{j}.png')
      
  first_image = Image.open(image_paths[0]).convert('RGB')
  other_images = [Image.open(image).convert('RGB') for image in image_paths[1:]]
  
  pdf_path = args.output_path + 'jeongganbo.pdf'
  first_image.save(pdf_path, save_all=True, append_images=other_images)