from pathlib import Path
from jngbomr import download_scorebooks 
from jngbomr import split_pdf_to_pages, repair_page_images
from jngbomr import split_page_to_jeonggans
from jngbomr import prepare_symbols

def main():
  jeongganbo_dir = Path().cwd() / 'dataset' / 'jeongganbo'
  
  # download_scorebooks(jeongganbo_dir)
  # split_pdf_to_pages(jeongganbo_dir)
  # repair_page_images(jeongganbo_dir, overwrite=True)
  # split_page_to_jeonggans(jeongganbo_dir)
  prepare_symbols(jeongganbo_dir)


if __name__ == '__main__':
  main()