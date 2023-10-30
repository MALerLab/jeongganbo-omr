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