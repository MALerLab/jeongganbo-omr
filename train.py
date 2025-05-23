import os, sys, getopt
import logging
import random
from pathlib import Path
import numpy as np
import hydra

from omegaconf import OmegaConf, DictConfig

import wandb

import torch
from torch.utils.data import DataLoader

from jngbomr import get_img_paths
from jngbomr import TransformerOMR
from jngbomr import CosineLRScheduler, Dataset, pad_collate, Trainer, get_nll_loss, LabelStudioDataset, draw_low_confidence_plot



@hydra.main(config_path='configs/', config_name='config')
def main(conf: DictConfig):
  # project dir setting
  time_prefix = '-'.join('-'.join(os.getcwd().split('/')[-2:])[2:].split('-')[:-1]) # YY-MM-DD-HH-MM
  os.mkdir(os.path.join(os.getcwd(), 'model'))
  original_wd = Path(hydra.utils.get_original_cwd())
  
  # random seed setting
  if conf.general.random_seed:
    torch.manual_seed(conf.general.random_seed)
    np.random.seed(conf.general.random_seed_synth)
    random.seed(conf.general.random_seed_synth)
  
  wandb_run = None
  
  if conf.wandb.do_log:
    wandb_run = wandb.init(
      project=conf.wandb.project,
      name=f'{time_prefix}_{conf.general.model_name}',
      entity=conf.wandb.entity,
      config=OmegaConf.to_container(conf, resolve=True, throw_on_missing=True),
      notes=f'{os.getcwd()}'
    )
  
  device = torch.device(conf.general.device)
  
  print('\ndata_set loading...')
  
  note_img_path_dict = get_img_paths(original_wd / 'dataset/jeongganbo/synth/', ['notes', 'symbols'])
  
  train_set = Dataset(original_wd /conf.data_path.train, note_img_path_dict, synth_config=dict(conf.synth))
  
  if conf.data_path.train_aux:
    aux_train_set = LabelStudioDataset(original_wd /conf.data_path.train_aux, original_wd / 'dataset/jeongganbo/split_jeonggans')
  
  valid_set_synthed = Dataset(original_wd /conf.data_path.valid_synthed, note_img_path_dict, is_valid=True)
  valid_set_HL = LabelStudioDataset(original_wd /conf.data_path.valid_HL, original_wd / 'dataset/jeongganbo/split_jeonggans', remove_borders=True, is_valid=True)
  
  if conf.data_path.test:
    test_set = LabelStudioDataset(original_wd / conf.data_path.test, original_wd / 'dataset/jeongganbo/split_jeonggans', remove_borders=conf.test_setting.remove_borders, is_valid=True)
  
  train_batch_size = conf.dataloader.batch_size
  aux_loader = None
  
  if conf.data_path.train_aux and conf.dataloader.mix_aux:
    aux_batch_size = int(conf.dataloader.aux_ratio * conf.dataloader.batch_size)
    train_batch_size = conf.dataloader.batch_size - aux_batch_size
    aux_loader = DataLoader(aux_train_set, batch_size=aux_batch_size, shuffle=True, collate_fn=pad_collate, num_workers=conf.dataloader.num_workers_load, drop_last=True)
  
  elif conf.data_path.train_aux and not conf.dataloader.mix_aux:
    aux_loader = DataLoader(aux_train_set, batch_size=train_batch_size, shuffle=True, collate_fn=pad_collate, num_workers=conf.dataloader.num_workers_load, drop_last=True)
  
  train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, collate_fn=pad_collate, num_workers=conf.dataloader.num_workers_synth)
  
  valid_synthed_loader = DataLoader(valid_set_synthed, batch_size=1000, shuffle=False, collate_fn=pad_collate, num_workers=conf.dataloader.num_workers_synth)
  valid_HL_loader = DataLoader(valid_set_HL, batch_size=500, shuffle=False, collate_fn=pad_collate, num_workers=conf.dataloader.num_workers_load)
  
  if conf.data_path.test:
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, collate_fn=pad_collate, num_workers=conf.dataloader.num_workers_load)
  
  print('COMPLETE: data_set loading')

  print('\nsaving tokenizer')
  tokenizer = train_set.tokenizer + valid_set_synthed.tokenizer + valid_set_HL.tokenizer + test_set.tokenizer
  
  if conf.data_path.train_aux:
    tokenizer += aux_train_set.tokenizer
    aux_train_set.tokenizer = tokenizer
  
  train_set.tokenizer = tokenizer
  valid_set_synthed.tokenizer = tokenizer
  valid_set_HL.tokenizer = tokenizer
  test_set.tokenizer = tokenizer
  
  with open(f'model/{conf.general.model_name}_tokenizer.txt', 'w') as f:
    f.write('\n'.join(tokenizer.vocab))
  
  print('COMPLETE: saving tokenizer')

  print('\nmodel initializing...')
  # model = OMRModel(80, vocab_size=len(tokenizer.vocab), num_gru_layers=2)
  model = TransformerOMR(conf.model.dim, len(tokenizer.vocab), enc_depth=conf.model.enc_depth, dec_depth=conf.model.dec_depth, num_heads=8, dropout=conf.model.dropout)
  optimizer = torch.optim.Adam(model.parameters(), lr=conf.model.lr)
  scheduler = CosineLRScheduler(optimizer, warmup_steps=1000, total_steps=len(train_loader), lr_min_ratio=0.0001, cycle_length=1.0)
  
  logger = logging.getLogger(__name__)
  logging.basicConfig(filename='best_checkpoint_log.csv', format='%(message)s', level=logging.INFO)

  trainer = Trainer(
    model, 
    optimizer, 
    get_nll_loss, 
    train_loader, 
    valid_synthed_loader, 
    tokenizer,
    scheduler=scheduler,
    aux_loader=aux_loader if aux_loader else None,
    aux_freq=conf.dataloader.aux_freq,
    mix_aux=conf.dataloader.mix_aux,
    aux_valid_loader=valid_HL_loader,
    device=device, 
    wandb=wandb_run, 
    model_name=conf.general.model_name,
    model_save_path='model',
    checkpoint_logger=logger
  )
  
  print('COMPLETE: model initializing')
  
  print('\nTraining...')
  
  trainer.train_and_validate()
  
  print('COMPELETE: Training')
  
  # post train logging
  print('\nStart: Post training logging - Last Synthed Validation')
  valid_acc, valid_metric_dict, valid_pred_list, valid_confi_list = trainer.validate(external_loader=valid_synthed_loader, with_confidence=True)
  valid_bottom_30 = draw_low_confidence_plot(valid_set_synthed, valid_confi_list, valid_pred_list)
  
  if conf.wandb.do_log:
    wandb_run.log({
      'Last synthed validation result': wandb.Table(columns=['valid_acc'] + list(valid_metric_dict.keys()), data=[[valid_acc] + list(valid_metric_dict.values())]),
      "Last synthed validation confidence Bottom-30": wandb.Image(valid_bottom_30, caption="valid synthed bottom-30")
    })
  print('COMPLETE: Post training logging - Last Synthed Validation')
  
  print('\nStart: Post training logging - Last HL Validation')
  valid_acc, valid_metric_dict, valid_pred_list, valid_confi_list = trainer.validate(external_loader=valid_HL_loader, with_confidence=True)
  valid_bottom_30 = draw_low_confidence_plot(valid_set_HL, valid_confi_list, valid_pred_list)
  
  if conf.wandb.do_log:
    wandb_run.log({
      'Last HL validation result': wandb.Table(columns=['valid_acc'] + list(valid_metric_dict.keys()), data=[[valid_acc] + list(valid_metric_dict.values())]),
      "Last HL validation confidence Bottom-30": wandb.Image(valid_bottom_30, caption="valid HL bottom-30")
    })
  print('COMPLETE: Post training logging - Last HL Validation')
  
  if conf.data_path.test:
    print('\nStart: Post training logging - Test')
    
    if conf.test_setting.with_best:
      test_model = TransformerOMR(conf.model.dim, len(tokenizer.vocab), enc_depth=conf.model.enc_depth, dec_depth=conf.model.dec_depth, num_heads=8, dropout=conf.model.dropout)
      
      test_model.load_state_dict(torch.load(f'model/{conf.general.model_name}_HL_{conf.test_setting.target_metric}_best.pt', map_location='cpu')['model'])
      
      tester = Trainer(
        test_model, 
        None, 
        None, 
        None, 
        test_loader, 
        tokenizer,
        scheduler=None,
        aux_loader=None,
        aux_freq=None,
        mix_aux=None,
        aux_valid_loader=None,
        device=device, 
        wandb=wandb_run, 
        model_name=conf.general.model_name,
        model_save_path='model',
        checkpoint_logger=None
      )
      
      test_acc, test_metric_dict, test_pred_list, test_confi_list = tester.validate(with_confidence=True)
      test_bot_30 = draw_low_confidence_plot(test_set, test_confi_list, test_pred_list)
    
    else:
      test_acc, test_metric_dict, test_pred_list, test_confi_list = trainer.validate(external_loader=test_loader, with_confidence=True)
      test_bot_30 = draw_low_confidence_plot(test_set, test_confi_list, test_pred_list)
    
    if conf.wandb.do_log:
      wandb_run.log({
        'Last test result': wandb.Table(columns=['test_acc'] + list(test_metric_dict.keys()), data=[[test_acc] + list(test_metric_dict.values())]),
        "Last test confidence Bottom-30": wandb.Image(test_bot_30, caption="test bottom-30")
      })
    
    if conf.wandb.is_sweep:
      wandb_run.log({
        'test_acc': test_acc
      })
    print('COMPLETE: Post training logging - Test')

if __name__ == "__main__":
  main()