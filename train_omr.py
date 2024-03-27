import sys, getopt
from pathlib import Path
import hydra

from omegaconf import OmegaConf, DictConfig

import wandb

import torch
from torch.utils.data import DataLoader

from exp_utils.jeonggan_synthesizer import get_img_paths
from exp_utils.model_zoo import OMRModel, TransformerOMR
from exp_utils.train_utils import CosineLRScheduler, Dataset, pad_collate, Trainer, get_nll_loss, LabelStudioDataset, draw_low_confidence_plot


def getConfs(argv):
  args = argv[1:]
  
  try:
    opt_list, _ = getopt.getopt(args, 'f:n:p:e:')
    
  except getopt.GetoptError:
    print('somethings gone wrong')
    return None

  opt_dict = {}
  
  for opt in opt_list:
    name, value = opt 
    name = name.replace('-', '')
    
    if value:
      opt_dict[name] = value
    else:
      opt_dict[name] = name
    
  conf = OmegaConf.load(opt_dict['f'])
  
  if opt_dict.get('e'):
    num_epoch_arg = int(opt_dict['e'])
    conf.num_epoch = num_epoch_arg
    
  if opt_dict.get('n'):
    name_arg = opt_dict['n']
    conf.run_name = name_arg
    
  if opt_dict.get('p'):
    project_arg = opt_dict['p']
    conf.project_name = project_arg
  
  return conf


@hydra.main(config_path='configs/', config_name='synth_only_240327')
def main(conf: DictConfig):
  # conf = getConfs(argv)
  original_wd = Path(hydra.utils.get_original_cwd())
  wandb_run = None
  
  if conf.wandb_log:
    wandb_run = wandb.init(
      project=conf.project_name,
      name=conf.model_name,
      entity=conf.wandb_entity,
      notes=conf.wandb_notes
    )
  
  device = torch.device(conf.device)
  
  print('\nSTART: data_set loading\n')
  
  note_img_path_dict = get_img_paths(original_wd / 'test/synth/src', ['notes', 'symbols'])
  
  train_set = Dataset(original_wd /conf.train_set_path, note_img_path_dict)
  valid_set = Dataset(original_wd /conf.valid_set_path, note_img_path_dict, is_valid=True)
  aux_train_set = LabelStudioDataset(original_wd /conf.aux_train_set_path, original_wd / 'jeongganbo-png/splited-pngs')
  test_set = LabelStudioDataset(original_wd / conf.test_set_path, original_wd / 'jeongganbo-png/splited-pngs')
  
  train_loader = DataLoader(train_set, batch_size=conf.train_batch_size, shuffle=True, collate_fn=pad_collate, num_workers=8)
  valid_loader = DataLoader(valid_set, batch_size=1000, shuffle=False, collate_fn=pad_collate, num_workers=4)
  aux_loader = DataLoader(aux_train_set, batch_size=conf.train_batch_size, shuffle=True, collate_fn=pad_collate, num_workers=4, drop_last=True)
  test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, collate_fn=pad_collate, num_workers=4)
  
  print('\nCOMPLETE: data_set loading\n')

  tokenizer = train_set.tokenizer + valid_set.tokenizer + aux_train_set.tokenizer + test_set.tokenizer
  train_set.tokenizer = tokenizer
  valid_set.tokenizer = tokenizer
  aux_train_set.tokenizer = tokenizer
  test_set.tokenizer = tokenizer
  
  with open(original_wd / f'model/{conf.model_name}_tokenizer.txt', 'w') as f:
    f.write('\n'.join(tokenizer.vocab))

  # model = OMRModel(80, vocab_size=len(tokenizer.vocab), num_gru_layers=2)
  model = TransformerOMR(128, len(tokenizer.vocab))
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  scheduler = CosineLRScheduler(optimizer, warmup_steps=1000, total_steps=len(train_loader), lr_min_ratio=0.0001, cycle_length=1.0)

  trainer = Trainer(model, 
                    optimizer, 
                    get_nll_loss, 
                    train_loader, 
                    valid_loader, 
                    tokenizer,
                    scheduler=scheduler,
                    aux_loader=aux_loader,
                    device=device, 
                    wandb=wandb_run, 
                    model_name=conf.model_name, 
                    model_save_path=original_wd / 'model')
  
  print('\nStart: Training\n')
  
  trainer.train_and_validate()
  
  print('\nCOMPELETE: Training')
  
  # post train logging
  print('\nStart: Post training logging - Validation\n')
  valid_acc, valid_metric_dict, valid_pred_list, valid_confi_list = trainer.validate(external_loader=valid_loader, with_confidence=True)
  valid_bottom_30 = draw_low_confidence_plot(valid_set, valid_confi_list, valid_pred_list)
  
  if conf.wandb_log:
    wandb_run.log({
      'Last validation result': wandb.Table(columns=['valid_acc'] + list(valid_metric_dict.keys()), data=[[valid_acc] + list(valid_metric_dict.values())]),
      "Last validation confidence Bottom-30": wandb.Image(valid_bottom_30, caption="valid bottom-30")
    })
  print('\nCOMPLETE: Post training logging - Validation\n')
  
  print('\nStart: Post training logging - Test\n')  
  test_acc, test_metric_dict, test_pred_list, test_confi_list = trainer.validate(external_loader=test_loader, with_confidence=True)
  test_bot_30 = draw_low_confidence_plot(test_set, test_confi_list, test_pred_list)
  
  if conf.wandb_log:
    wandb_run.log({
      'Last test result': wandb.Table(columns=['test_acc'] + list(test_metric_dict.keys()), data=[[test_acc] + list(test_metric_dict.values())]),
      "Last test confidence Bottom-30": wandb.Image(test_bot_30, caption="test bottom-30")
    })
  print('\nCOMPLETE: Post training logging - Test\n')
  
  print('\nCOMPLETE')

if __name__ == "__main__":
  main()
