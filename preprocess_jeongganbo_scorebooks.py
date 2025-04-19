from pathlib import Path
from jngbomr import download_scorebooks 
from jngbomr import split_pdf_to_pages, repair_page_images
from jngbomr import split_page_to_jeonggans

def main():
  jeongganbo_images_dir = Path().cwd() / 'dataset' / 'jeongganbo_images'
  
  # download_scorebooks(jeongganbo_images_dir)
  split_pdf_to_pages(jeongganbo_images_dir)
  repair_page_images(jeongganbo_images_dir, overwrite=True)
  split_page_to_jeonggans(jeongganbo_images_dir)


if __name__ == '__main__':
  main()