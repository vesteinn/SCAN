import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import generate_scan_dictionary, SCANDataset


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, dictionary):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dictionary = dictionary

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, input, hidden):
        # TODO: make batched
        embedded = self.embedding(input)
        embedded = embedded.view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden


# TODO: add attention
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, dictionary):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.dictionary = dictionary

        self.embedding = nn.Embedding(len(dictionary), hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, len(dictionary))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


class LSTMRNN(nn.Module):
    # TODO: make dictionary class and write there
    EOS = "EOS"
    BOS = "BOS"

    def __init__(self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        src_dictionary,
        tgt_dictionary,
        max_length=64):
        super(LSTMRNN, self).__init__()

        self.input_size = input_size
        self.max_length = 64
        self.num_layers = num_layers
        
        # TODO: is this the same dropout as referenced in the paper?
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout, src_dictionary)
        self.decoder = LSTMDecoder(hidden_size, num_layers, dropout, tgt_dictionary)

    def device(self):
        # TODO: consider optimizing
        return next(self.encoder.parameters()).device

    def init_hidden(self):
        # TODO: make it work for bszs
        hidden = self.num_layers * [torch.zeros(2, 1, self.encoder.hidden_size, device=self.device(), requires_grad=True)]
        return hidden

    def forward(self, input, target, teacher_forcing=False):
        input_length = input.shape[0]
        target_length = target.shape[0]

        # Store state for encoder steps
        encoder_outputs = torch.zeros(
            self.max_length,
            self.encoder.hidden_size,
            device=self.device()
        )
        encoder_hidden = self.init_hidden()
        for idx in range(input_length):
            # Note: No need to loop with torch LSTM, but needed fro GRU?
            encoder_output, encoder_hidden = self.encoder(
                input[idx],
                encoder_hidden
            )
            encoder_outputs[idx] = encoder_output.squeeze()

        # TODO: check that BOS is not already on data using teacher forcing
        decoder_input = torch.tensor(
            self.decoder.dictionary[self.BOS],
            device=self.device()
        )
        decoder_hidden = encoder_hidden

        decoder_outputs = []

        for idx in range(target_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            decoder_outputs.append(decoder_output)
            if teacher_forcing:
                decoder_input = target[idx]  # Teacher forcing
            else:
               _topv, topi = decoder_output.topk(1)
               decoder_input = topi.squeeze().clone().detach()  # detach from history as input
            
            if decoder_input.item() == self.decoder.dictionary[self.EOS]:
               break

        # Remove BOS
        return torch.stack(decoder_outputs) 


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, dictionary):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dictionary = dictionary

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size,num_layers,dropout=dropout)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, dictionary):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.dictionary = dictionary

        self.embedding = nn.Embedding(len(dictionary), hidden_size)

        self.gru = nn.GRU(hidden_size,hidden_size,num_layers,dropout=dropout)
        self.out = nn.Linear(hidden_size, len(dictionary))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class AttnGRUDecoder(nn.Module):
    def __init__(self,hidden_size, num_layers,dropout,dictionary):
        super(AttnGRUDecoder,self).__init__()
        self.hidden_size = hidden_size
        self.dictionary = dictionary
        self.dropout = dropout
        self.max_length = 64

        self.embedding = nn.Embedding(len(dictionary), hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)

        self.gru = nn.GRU(hidden_size,hidden_size,num_layers,dropout=dropout)
        self.out = nn.Linear(hidden_size, len(dictionary))
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(self.attn(torch.cat([embedded[0],hidden[0]],1)), dim = 1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        
        output = torch.cat([embedded[0],attn_applied[0]],1)
        output = self.attn_combine(output).unsqueeze(0)
        
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]),dim=1)
        return output, hidden,attn_weights
        
        
        
        
        
        
        
class GRURNN(nn.Module):

    EOS = "EOS"
    BOS = "BOS"
    
    def __init__(self,input_size,hidden_size,num_layers,dropout,src_dictionary,tgt_dictionary,max_length=64):
        super(GRURNN,self).__init__()
        
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_length = 64
        
        self.encoder = GRUEncoder(input_size, hidden_size, num_layers, dropout, src_dictionary)
        self.decoder = AttnGRUDecoder(hidden_size, num_layers, dropout, tgt_dictionary)
    
    
    def device(self):
        return next(self.encoder.parameters()).device
    
    def init_hidden(self):
        hidden = torch.zeros(1, 1, self.encoder.hidden_size, device=self.device(), requires_grad=True)
        return hidden



    def forward(self, input, target, teacher_forcing = False):
        
        input_length = input.shape[0]
        target_length = target.shape[0]
    
        encoder_outputs = torch.zeros(
            self.max_length,
            self.encoder.hidden_size,
            device=self.device()
        )
        encoder_hidden = self.init_hidden()
        for idx in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input[idx],
                encoder_hidden
            )
            encoder_outputs[idx] = encoder_output.squeeze()
    
        decoder_input = torch.tensor(
            self.decoder.dictionary[self.BOS],
            device=self.device()
        )
        decoder_hidden = encoder_hidden
    
        decoder_outputs = []
    
        for idx in range(target_length):
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            decoder_outputs.append(decoder_output)
            if teacher_forcing:
                decoder_input = target[idx]  # Teacher forcing
            else:
                _topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().clone().detach()  # detach from history as input
            
            if decoder_input.item() == self.decoder.dictionary[self.EOS]:
                break
        return torch.stack(decoder_outputs) 



if __name__ == "__main__":
    tasks = "../../data/SCAN/tasks.txt"
    src_dict, tgt_dict = generate_scan_dictionary(tasks, add_bos=True, add_eos=True)
    print("Dictionary loaded")
    dataset = SCANDataset(tasks, src_dict, tgt_dict)
    print("Dataset loaded")
    model = LSTMRNN(
        len(src_dict),
        100,
        10,
        2,
        0.5,
        src_dict,
        tgt_dict
    )
    print("LSTM model initialized")
    gru_model = GRURNN(
        len(src_dict),
        50,
        1,
        0.5,
        src_dict,
        tgt_dict
    )
    print("GRU Model initialized")
    input, target = dataset[0]
    print(f"Input shape {input.shape}. Target shape {target.shape}.")
    outputs = model.forward(input, target, teacher_forcing=False)
    print("LSTM model forward ran without teacher forcing")
    outputs = model.forward(input, target, teacher_forcing=True)
    print("LSTM model forward ran with teacher forcing")
    outputs = gru_model.forward(input, target, teacher_forcing=False)
    print("GRU model forward ran without teacher forcing")
    outputs = gru_model.forward(input, target, teacher_forcing=True)
    print("GRU model forward ran with teacher forcing")
