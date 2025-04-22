from .jeonggan_utils import JeongganProcessor, JeongganSynthesizer
from .jeonggan_utils import get_img_paths
from .jeonggan_utils import JeongganboPageDrawer, JeongganboReader, Piece

from .jeonggan_utils import PIECE, START_PAGE
from .jeonggan_utils import NAME_EN_TO_KR, NAME_KR_TO_EN, NOTE_W_DUR_EN_SET, PNAME_LIST, PNAME_EN_LIST, SPECIAL_CHAR_TO_NAME, PNAME_EN_TO_KR, PNAME_KR_TO_EN, SYMBOL_W_DUR_LIST, SYMBOL_W_DUR_EN_LIST, SYMBOL_WO_DUR_LIST, SYMBOL_WO_DUR_EN_LIST, SYMBOL_WO_DUR_ADD_EN_LIST, SYMBOL_LIST, SYMBOL_EN_LIST, SYMBOL_W_DUR_EN_TO_KR, SYMBOL_W_DUR_KR_TO_EN, SYMBOL_WO_DUR_EN_TO_KR, SYMBOL_WO_DUR_KR_TO_EN

from .preprocess_utils import download_scorebooks, split_pdf_to_pages, repair_page_images, split_page_to_jeonggans, prepare_symbols

from .data_utils import pad_collate
from .data_utils import RandomBoundaryDrop, Dataset, LabelStudioDataset

from .vocab_utils import Tokenizer

from .model_zoo import TransformerOMR

from .trainer import Trainer

from .train_utils import get_nll_loss, CosineLRScheduler, draw_low_confidence_plot

from .inferencer import Inferencer, INFERENCER_DEFAULT_KWARGS


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