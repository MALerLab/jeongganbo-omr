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