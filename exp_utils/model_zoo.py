


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from x_transformers import Encoder, Decoder
from x_transformers.x_transformers import AbsolutePositionalEmbedding

from exp_utils import JeongganSynthesizer, PNAME_EN_LIST, SYMBOL_W_DUR_EN_LIST, JeongganProcessor
from exp_utils.jeonggan_synthesizer import get_img_paths


class OMRModel(nn.Module):
  def __init__(self, hidden_size, vocab_size, num_gru_layers=2):
    super().__init__()

    self.layers = nn.Sequential(
      ConvBlock(1, hidden_size//4, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size//4, hidden_size//4, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size//4, hidden_size//2, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size//2, hidden_size//2, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size//2, hidden_size, 3, 1, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(hidden_size, hidden_size, 3, 1, 1),
    )

    self.context_attention = ContextAttention(hidden_size, 4)
    self.cont2hidden = nn.Linear(hidden_size, hidden_size*num_gru_layers)
    self.cnn_gru = nn.GRU(hidden_size, hidden_size//2, 1, batch_first=True, dropout = 0.2, bidirectional=True)

    self.emb = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, num_gru_layers, batch_first=True, dropout = 0.2)

    self.attn_layer1 = QKVAttention(hidden_size)
    self.final_gru1 = nn.GRU(hidden_size * 2, hidden_size, 1, batch_first=True)

    self.attn_layer2 = QKVAttention(hidden_size)
    self.final_gru2 = nn.GRU(hidden_size * 2, hidden_size*2, 1, batch_first=True)
    

    self.proj = nn.Linear(hidden_size*2, vocab_size)

  def run_img_cnn(self, x):
    x = self.layers(x)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    x = x.contiguous()
    x, _ = self.cnn_gru(x)

    context_vector = self.context_attention(x)
    context_vector = self.cont2hidden(context_vector.relu())
    context_vector = context_vector.reshape(x.shape[0], -1, context_vector.shape[-1]//2).permute(1,0,2)
    context_vector = context_vector.contiguous()

    return x, context_vector

  def forward(self, x, y):
    x, context_vector = self.run_img_cnn(x)
    y = self.emb(y)
    gru_out, _ = self.gru(y, context_vector)
    attention = self.attn_layer1(gru_out, x)
    cat_out = torch.cat([gru_out, attention], dim=-1)
    cat_out, _ = self.final_gru1(cat_out)
    
    attention = self.attn_layer2(cat_out, x)
    cat_out = torch.cat([cat_out, attention], dim=-1)
    cat_out, _ = self.final_gru2(cat_out)

    logit = self.proj(cat_out)

    return logit

  @torch.inference_mode()  
  def inference(self, x, max_len=None, batch=True):
    # assert x.shape[0] == 1 # batch size must be 1

    x, last_hidden = self.run_img_cnn(x)
    
    y = torch.ones((x.shape[0], 1), dtype=torch.long).to(x.device)
    outputs = torch.ones_like(y) if batch else []
    
    final_gru_last_hidden1 = None
    final_gru_last_hidden2 = None
    
    max_len = max_len if max_len else 100
    
    for _ in range(max_len):
      y = self.emb(y)
      gru_out, last_hidden = self.gru(y, last_hidden)
      
      attention = self.attn_layer1(gru_out, x)
      cat_out = torch.cat([gru_out, attention], dim=-1)
      cat_out, final_gru_last_hidden1 = self.final_gru1(cat_out, final_gru_last_hidden1)        
      
      attention = self.attn_layer2(cat_out, x)
      cat_out = torch.cat([cat_out, attention], dim=-1)
      cat_out, final_gru_last_hidden2 = self.final_gru2(cat_out, final_gru_last_hidden2)

      logit = self.proj(cat_out)
      y = torch.argmax(logit, dim=-1)
      
      if batch:
        outputs = torch.cat([outputs, y], dim=1) 
      else:
        if y.item() == 2:
          break
        outputs.append(y.item())
    
    return outputs


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


class ContextAttention(nn.Module):
  def __init__(self, size, num_head):
    super(ContextAttention, self).__init__()
    self.attention_net = nn.Linear(size, size)
    self.num_head = num_head

    if size % num_head != 0:
      raise ValueError("size must be dividable by num_head", size, num_head)
    self.head_size = int(size/num_head)
    self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
    nn.init.uniform_(self.context_vector, a=-1, b=1)

  def get_attention(self, x):
    attention = self.attention_net(x)
    attention_tanh = torch.tanh(attention)
    attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
    similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
    similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
    return similarity

  def forward(self, x):
    attention = self.attention_net(x)
    attention_tanh = torch.tanh(attention)
    if self.head_size != 1:
      attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
      similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
      similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
      similarity[x.sum(-1)==0] = -1e10 # mask out zero padded_ones
      softmax_weight = torch.softmax(similarity, dim=1)

      x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
      weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1,1,1, x_split.shape[-1])
      attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
    else:
      softmax_weight = torch.softmax(attention, dim=1)
      attention = softmax_weight * x

    sum_attention = torch.sum(attention, dim=1)
    return sum_attention


class QKVAttention(nn.Module):
  def __init__(self, hidden_size, num_head=4, dropout=0.1):
    super().__init__()
    
    self.kv = nn.Linear(hidden_size, hidden_size*2)
    self.q = nn.Linear(hidden_size, hidden_size)
    self.num_head = num_head
    self.mlp = nn.Sequential(
      nn.Linear(hidden_size, hidden_size * 4),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_size * 4, hidden_size),
    )    
    
  def forward(self, _q, _kv):
    kv = self.kv(_kv)
    k, v = torch.split(kv, kv.shape[-1]//2, dim=-1)
    q = self.q(_q)
    attention_score = torch.bmm(q, k.permute(0,2,1))
    attention_score = torch.softmax(attention_score, dim=-1)
    attention = torch.bmm(attention_score, v)
    attention = self.mlp(attention)
    
    return attention
  


class TransformerOMR(nn.Module):
  def __init__(self, dim, vocab_size, dropout=0.1) -> None:
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
    
    self.encoder = Encoder(dim=dim, depth=2, heads=8, attn_dropout=dropout, ff_dropout=dropout)
    self.encoder_pos_enc = AbsolutePositionalEmbedding(dim, 100)
    self.dec_embedder = nn.Embedding(vocab_size, dim)
    self.decoder = Decoder(dim=dim, depth=6, heads=8, attn_dropout=dropout, ff_dropout=dropout, cross_attn_dropout=dropout, cross_attend=True)
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
  def inference(self, x, max_len=None, batch=True):
    x = self.run_img_cnn(x)
    x += self.encoder_pos_enc(x)
    x = self.encoder(x)
        
    dec_tokens = torch.ones((x.shape[0], 1), dtype=torch.long).to(x.device)
    is_ended = torch.zeros((x.shape[0]), dtype=torch.bool).to(x.device)
    max_len = max_len if max_len else 50
    
    for _ in range(max_len+1):
      y = self.dec_embedder(dec_tokens) + self.decoder_pos_enc(dec_tokens)
      y = self.decoder(y, x)[:, -1:]
      logit = self.proj(y)
      new_token = torch.argmax(logit, dim=-1)
      is_ended += (new_token[:,0] == 2)
      dec_tokens = torch.cat([dec_tokens, new_token], dim=1)
      if is_ended.all():
        break
    
    if dec_tokens.shape[1] < max_len+1:
      dec_tokens = torch.cat([dec_tokens, torch.zeros((dec_tokens.shape[0], max_len - dec_tokens.shape[1]+1), dtype=torch.long).to(x.device)], dim=1)

    return dec_tokens