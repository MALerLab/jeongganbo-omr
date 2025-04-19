from pathlib import Path
from jngbomr import download_scorebooks, split_pdf_to_pages, split_page_to_jeonggans

def main():
  jeongganbo_images_dir = Path().cwd() / 'dataset' / 'jeongganbo_images'
  
  download_scorebooks(jeongganbo_images_dir)
  split_pdf_to_pages(jeongganbo_images_dir)
  split_page_to_jeonggans(jeongganbo_images_dir)


if __name__ == '__main__':
  main()