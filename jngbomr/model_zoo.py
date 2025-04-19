import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from x_transformers import Encoder, Decoder
from x_transformers.x_transformers import AbsolutePositionalEmbedding

from .jeonggan_utils import JeongganProcessor, JeongganSynthesizer, get_img_paths 
from .jeonggan_utils.const import PNAME_EN_LIST, SYMBOL_W_DUR_EN_LIST


class TransformerOMR(nn.Module):
  def __init__(self, dim, vocab_size, enc_depth=2, dec_depth=6, num_heads=8, dropout=0.1) -> None:
    super().__init__()
    self.layers = nn.Sequential(
      ConvBlock(1, dim//4, 3, 1, 1),
      nn.Dropout(dropout),
      nn.MaxPool2d(2, 2),
      ConvBlock(dim//4, dim//4, 3, 1, 1),
      nn.Dropout(dropout),
      nn.MaxPool2d(2, 2),
      ConvBlock(dim//4, dim//2, 3, 1, 1),
      nn.Dropout(dropout),
      nn.MaxPool2d(2, 2),
      ConvBlock(dim//2, dim//2, 3, 1, 1),
      nn.Dropout(dropout),
      nn.MaxPool2d(2, 2),
      ConvBlock(dim//2, dim, 3, 1, 1),
      nn.Dropout(dropout),
      nn.MaxPool2d(2, 2),
      ConvBlock(dim, dim, 3, 1, 1),
    )
    
    self.encoder = Encoder(dim=dim, depth=enc_depth, heads=num_heads, attn_dropout=dropout, ff_dropout=dropout)
    self.encoder_pos_enc = AbsolutePositionalEmbedding(dim, 100)
    self.dec_embedder = nn.Embedding(vocab_size, dim)
    self.decoder = Decoder(dim=dim, depth=dec_depth, heads=num_heads, attn_dropout=dropout, ff_dropout=dropout, cross_attn_dropout=dropout, cross_attend=True)
    self.decoder_pos_enc = AbsolutePositionalEmbedding(dim, 100)
    self.proj = nn.Linear(dim, vocab_size)
  

  def run_img_cnn(self, x):
    x = self.layers(x)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    return x


  def forward(self, x, y):
    x = self.run_img_cnn(x)
    x += self.encoder_pos_enc(x)
    x = self.encoder(x)
    y = self.dec_embedder(y) + self.decoder_pos_enc(y)
    y = self.decoder(y, x)
    logit = self.proj(y)
    return logit
  

  @torch.inference_mode()  
  def inference(self, x, max_len=None, batch=True, return_confidence=False):
    x = self.run_img_cnn(x)
    x += self.encoder_pos_enc(x)
    x = self.encoder(x)
        
    dec_tokens = torch.ones((x.shape[0], 1), dtype=torch.long).to(x.device)
    is_ended = torch.zeros((x.shape[0]), dtype=torch.bool).to(x.device)
    max_len = max_len if max_len else 50
    if return_confidence: 
      confidence = torch.ones((x.shape[0])).to(x.device)
    
    for _ in range(max_len):
      y = self.dec_embedder(dec_tokens) + self.decoder_pos_enc(dec_tokens)
      y = self.decoder(y, x)[:, -1:]
      logit = self.proj(y)
      new_token = torch.argmax(logit, dim=-1)
      is_ended += (new_token[:,0] == 2)
      dec_tokens = torch.cat([dec_tokens, new_token], dim=1)
      if return_confidence:
        prob = torch.max(torch.softmax(logit, dim=-1)[:, 0],dim=1)[0]
        conf_clone = confidence.clone()
        conf_clone[~is_ended] *= prob[~is_ended]
        confidence = conf_clone
      if is_ended.all():
        break
    
    if return_confidence:
      return dec_tokens, confidence
    
    return dec_tokens



class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.GELU()

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x