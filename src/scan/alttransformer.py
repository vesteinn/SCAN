import os
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import sys

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Dataset from repo
from data import SCANDataset, generate_scan_dictionary
cur_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{cur_path}/../../data/SCAN"
tasks = f"{data_path}/tasks.txt"
src_dict, tgt_dict = generate_scan_dictionary(tasks, add_bos=True, add_eos=True)

rev_src_dict = {v: k for k, v in src_dict.items()}
rev_tgt_dict = {v: k for k, v in tgt_dict.items()}

# simple split
# train_file = "../../data/SCAN/simple_split/tasks_train_simple.txt"
# valid_file = "../../data/SCAN/simple_split/tasks_test_simple.txt"

train_file = "../../data/SCAN/length_split/tasks_train_length.txt" 
valid_file = "../../data/SCAN/length_split/tasks_test_length.txt"

train_file = sys.argv[1]
valid_file = sys.argv[2]

train_dataset = SCANDataset(train_file, src_dict, tgt_dict, device=DEVICE)
valid_dataset = SCANDataset(valid_file, src_dict, tgt_dict, device=DEVICE)

SRC_PAD_IDX = src_dict["PAD"]
TGT_PAD_IDX = tgt_dict["PAD"]
EOS_IDX = tgt_dict["EOS"]
BOS_IDX = tgt_dict["BOS"]

torch.manual_seed(int(sys.argv[6]))

SRC_VOCAB_SIZE = len(src_dict)
TGT_VOCAB_SIZE = len(tgt_dict)
#EMB_SIZE = 512
#NHEAD = 8
#FFN_HID_DIM = 512
#BATCH_SIZE = 128
#NUM_ENCODER_LAYERS = 3
#NUM_DECODER_LAYERS = 3

EMB_SIZE = int(sys.argv[3]) #100
NHEAD = 4
FFN_HID_DIM = int(sys.argv[3]) #100
NUM_ENCODER_LAYERS = int(sys.argv[4]) #1
NUM_DECODER_LAYERS = int(sys.argv[4]) #1

BATCH_SIZE = int(sys.argv[5]) #1
LEARNING_RATE = float(sys.argv[7])

print(f"Starting run on {train_file} and {valid_file}")
print(f"BSZ {BATCH_SIZE}")
print(f"EMB_size {EMB_SIZE}")
print(f"NHEAD {NHEAD}")
print(f"NUM_LAYERS {NUM_ENCODER_LAYERS}")

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.0):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == SRC_PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == TGT_PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
#optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


from torch.nn.utils.rnn import pad_sequence


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
#text_transform = {}
#for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
#    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
#                                               vocab_transform[ln], #Numericalization
#                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=SRC_PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=TGT_PAD_IDX)
    return src_batch, tgt_batch


from torch.utils.data import DataLoader

import tqdm

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    #train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in tqdm.tqdm(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


from timeit import default_timer as timer
NUM_EPOCHS = 100000 // len(train_dataset) 


for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src, tgt):
    model.eval()
    src = src.view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=50, start_symbol=BOS_IDX).flatten()
    
    src_str = " ".join(rev_src_dict[t.item()] for t in src)
    tgt_str = " ".join(rev_tgt_dict[t.item()] for t in tgt) 

    #return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("BOS", "").replace("EOS", "")
    return tgt_tokens, src_str, tgt_str


def final_eval(model):
    accuracy = []
    for idx, data in enumerate(valid_dataset):
        src, tgt = data
        trans_tgt, src_str, tgt_str = translate(model, src, tgt)
        acc = False
        if trans_tgt.shape[0] != tgt.shape[0]:
            acc = False
        elif torch.all(torch.eq(trans_tgt, tgt)):
            acc = True
        accuracy.append(acc)
        cur_acc = 100 * sum(accuracy) / len(accuracy)
        if idx % 10 == 0:
            print(f"{idx}: {cur_acc:0.2f} %")
            print(f"src: {src}")
            print(f"tgt: {tgt}")
            print(f"trans_tgt: {trans_tgt}")
    print(f"Final accuracy {cur_acc:0.2f} %")

final_eval(transformer)
