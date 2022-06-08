import torch
import torch.nn as nn
import torch.optim as optim

import utils
import imp
imp.reload(utils)

import torchtext
from torchtext.legacy.datasets import text_classification
from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator

# import spacy

import random
import math
import time
import numpy as np
import tqdm

from nltk.tokenize import WordPunctTokenizer
from nltk.translate.bleu_score import corpus_bleu
from transformers import get_linear_schedule_with_warmup
import transformer


import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import clear_output



tokenizer_W = WordPunctTokenizer()

def tokenize(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())

def process_dataset(path_do_data, bf=False):
  SRC = Field(tokenize=tokenize,
              init_token = '<sos>', 
              eos_token = '<eos>', 
              lower = True,
              batch_first = bf)

  TRG = Field(tokenize=tokenize,
              init_token = '<sos>', 
              eos_token = '<eos>', 
              lower = True,
              batch_first = bf)

  dataset = torchtext.legacy.data.TabularDataset(
      path=path_do_data,
      format='tsv',
      fields=[('trg', TRG), ('src', SRC)]
  )    

  train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])

  print(f"Number of training examples: {len(train_data.examples)}")
  print(f"Number of validation examples: {len(valid_data.examples)}")
  print(f"Number of testing examples: {len(test_data.examples)}")

  SRC.build_vocab(train_data, min_freq = 3)
  TRG.build_vocab(train_data, min_freq = 3)

  print(f"Unique tokens in source (ru) vocabulary: {len(SRC.vocab)}")
  print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

  return train_data, valid_data, test_data, SRC, TRG


def _len_sort_key(x):
    return len(x.src)


def transformer_init(Encoder, Decoder, Seq2Seq, SRC, TRG, device):
  print(device)

  INPUT_DIM = len(SRC.vocab)
  OUTPUT_DIM = len(TRG.vocab)
  HID_DIM = 256
  ENC_LAYERS = 2
  DEC_LAYERS = 2
  ENC_HEADS = 8
  DEC_HEADS = 8
  ENC_PF_DIM = 512
  DEC_PF_DIM = 512
  ENC_DROPOUT = 0.2
  DEC_DROPOUT = 0.2

  enc = Encoder(INPUT_DIM, 
                HID_DIM, 
                ENC_LAYERS, 
                ENC_HEADS, 
                ENC_PF_DIM, 
                ENC_DROPOUT, 
                device)

  dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                device)

  SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
  TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

  model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
  model.apply(init_weights_xavier)
  print(f'The model has {count_parameters(model):,} trainable parameters')
  return model  

def model_init(Encoder, Decoder, Seq2Seq, SRC, TRG, device, attention = None):

  print(device)

  INPUT_DIM = len(SRC.vocab)
  OUTPUT_DIM = len(TRG.vocab)
  ENC_EMB_DIM = 256
  DEC_EMB_DIM = 256
  HID_DIM = 512
  N_LAYERS = 2
  ENC_DROPOUT = 0.3
  DEC_DROPOUT = 0.3
  if attention:
    attn = attention(HID_DIM, HID_DIM,)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, DEC_DROPOUT, attn)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, ENC_DROPOUT)
  else:
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)

  model = Seq2Seq(enc, dec, device).to(device)
  model.apply(init_weights)

  print(f'The model has {count_parameters(model):,} trainable parameters')
  return model

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)
        
def init_weights_xavier(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None, transformer=False):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        if transformer:
          
          output, _ = model(src, trg[:,:-1])
          output = output.contiguous().view(-1, output.shape[-1])
          trg = trg[:,1:].contiguous().view(-1)
        else:
          output = model(src, trg)         
         
          output = output[1:].view(-1, output.shape[-1])
          trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        history.append(loss.cpu().data.numpy())
        if (i+1)%10==0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            
            plt.show()
     
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, TRG, transformer=False):
    
    model.eval()
    
    epoch_loss = 0
    
    history = []
    bleu_history = []
    original_text = []
    generated_text = []
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            imp.reload(utils)
            src = batch.src
            trg = batch.trg
            if transformer:
                output, _ = model(src, trg[:,:-1])
                output_for_bleu = output
                output = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:,1:].contiguous().view(-1)
                output_for_bleu = output_for_bleu.argmax(dim=2)
            
                generated_text.extend([utils.get_text(x, TRG.vocab) for x in output_for_bleu.cpu().numpy()])
                original_text.extend([utils.get_text(x, TRG.vocab) for x in batch.trg])

            else:
                output = model(src, trg, 0) #turn off teacher forcing
                output_bl = output.argmax(dim=-1)
            
                original_text.extend([utils.get_text(x, TRG.vocab) for x in trg.cpu().numpy().T])
                generated_text.extend([utils.get_text(x, TRG.vocab) for x in output_bl[1:].detach().cpu().numpy().T])

                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    score = corpus_bleu([[text] for text in original_text], generated_text) * 100    
    return epoch_loss / len(iterator), score


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_loop(model, model_name, device, train_data, valid_data, test_data, SRC, TRG, train_history=None, valid_history=None, N_EPOCHS = 10, transformer=False):
     
    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE, 
        device = device,
        sort_key=_len_sort_key
    )
    PAD_IDX = TRG.vocab.stoi['<pad>']
    optimizer = optim.Adam(model.parameters())
    # scheduler = get_linear_schedule_with_warmup(
    #             optimizer,
    #             num_warmup_steps=0,
    #             num_training_steps=N_EPOCHS
    #         )
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

    train_history = []
    valid_history = []
    bleu_history = []
    CLIP = 1

    best_valid_loss = float('inf')   

    for epoch in range(N_EPOCHS):
    
      start_time = time.time()
      
      train_loss = train(model, train_iterator, optimizer, criterion, CLIP, train_history, valid_history, transformer)
      # scheduler.step()
      valid_loss, score = evaluate(model, valid_iterator, criterion, TRG, transformer)
      
      end_time = time.time()
      
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      
      if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          torch.save(model.state_dict(), f'{model_name}.pt')
      
      train_history.append(train_loss)
      valid_history.append(valid_loss)
      bleu_history.append(score)
      print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    return model, test_iterator, bleu_history


def sample_and_bleu(model, test_iterator, test_data, TRG, device, idxs=[1,7400,239,6245,2390,4800], transformer=False):
  
  import imp
  imp.reload(utils)
  get_text = utils.get_text

  original_text = []
  generated_text = []
  losses = []
  PAD_IDX = TRG.vocab.stoi['<pad>']
  criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
  model.eval()
  epoch_loss = 0
  with torch.no_grad():

      for i, batch in tqdm.tqdm(enumerate(test_iterator)):
                    
          src = batch.src
          trg = batch.trg

          if transformer:
            output, _ = model(src, trg[:, :-1])
            out_for_loss = output.contiguous().view(-1, output.shape[-1])
            output = output.argmax(dim=2)
            
            generated_text.extend([get_text(x, TRG.vocab) for x in output.cpu().numpy()])
            original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy()])

            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(out_for_loss, trg)
            losses.append(loss.item())
                
          else:
            
            output = model(src, trg, 0) 
            out_for_loss = output

            output = output.argmax(dim=-1)
            original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, TRG.vocab) for x in output[1:].detach().cpu().numpy().T])
            out_for_loss = out_for_loss[1:].view(-1, out_for_loss.shape[-1])
            
            trg = trg[1:].view(-1)
            loss = criterion(out_for_loss, trg)
            losses.append(loss.item())
          epoch_loss += loss.item()
  test_loss = epoch_loss/len(test_iterator)
  score = corpus_bleu([[text] for text in original_text], generated_text) * 100

  print ("BLEU: ", score, "\n")
  print("-------")
  print("Примеры сгенерированных переводов:")
  for idx in idxs:
    print('Original: {}'.format(' '.join(original_text[idx])))
    print('Generated: {}'.format(' '.join(generated_text[idx])))
    print()
 
  
  best = np.argsort(np.array(losses))[:3]
  worst = np.argsort(np.array(losses))[-3:]
  print("-------")
  print("Удачные переводы")
  for i in best:
    sample = len(batch) * i + 2
    print("Original:", " ".join(original_text[sample]))
    print("Generated:", " ".join(generated_text[sample]))
    print()
  print("-------")
  print("Неудачные переводы")
  for i in worst:
    sample = len(batch) * i + 2
    print("Original:", " ".join(original_text[sample]))
    print("Generated:", " ".join(generated_text[sample]))  
    print()


  batch32_iterator = torchtext.legacy.data.Iterator(test_data, batch_size=32, shuffle=True, device=device)  
  start_time = time.time()
  with torch.no_grad():
    
    generated_text=[]
    for i, batch in enumerate(batch32_iterator):
                    
        src = batch.src
        trg = batch.trg
        if transformer:
          output, _ = model(src, trg[:, :-1])
          out_for_loss = output.contiguous().view(-1, output.shape[-1])
          trg_pred = output.argmax(dim=2).cpu().numpy()
                          
          generated_text.append([get_text(x, TRG.vocab) for x in trg_pred])
        else:      
          output = model(src, trg, 0)
          output = output.argmax(dim=-1)              
          generated_text.extend([utils.get_text(x, TRG.vocab) for x in output[1:].detach().cpu().numpy().T])
        end_time = time.time()
        break
  print("Время инференса батча 32: ", round(end_time - start_time, 4))
  return score, test_loss
