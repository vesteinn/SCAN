import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import generate_scan_dictionary, SCANDataset


#
# TODO: The dropout used here is not same as in paper!
#       LSTM/GRU do not support dropout with a single layer.
#       We need a drop out layer after also !
# 

class Encoder(nn.Module):
    def _get_hidden_type(self):
        raise NotImplementedError

    def __init__(self, input_size, hidden_size, num_layers, dropout, dictionary):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.dictionary = dictionary

        self.embedding = nn.Embedding(input_size, hidden_size)
        layer_type = self._get_hidden_type()
        self.hidden_layers = layer_type(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, input, hidden):
        # TODO: make batched
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.hidden_layers(embedded, hidden)
        return output, hidden


class Decoder(nn.Module):
    def _get_hidden_type(self, *args):
        raise NotImplementedError

    def __init__(self, hidden_size, num_layers, dropout, dictionary, use_attention=False, max_length=64):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.dictionary = dictionary
        self.max_length = max_length
        self.use_attention = use_attention
        self.embedding = nn.Embedding(len(dictionary), hidden_size)

        if use_attention:
            self.attn = nn.Linear(self.hidden_size*2, self.max_length)
            self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.dropout = nn.Dropout(dropout)

        layer_type = self._get_hidden_type()
        self.hidden_layers = layer_type(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, len(dictionary))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        attn_weights = None
        if self.use_attention:
            # Should this be here? Or always/never?
            embedded = self.dropout(embedded)
            attn_weights = F.softmax(self.attn(torch.cat([embedded[0],hidden[0]],1)), dim = 1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
            output = torch.cat([embedded[0],attn_applied[0]],1)
            output = self.attn_combine(output).unsqueeze(0)
            output, hidden = self.hidden_layers(output, hidden)
            output = F.log_softmax(self.out(output[0]),dim=1)
        else:
            output, hidden = self.hidden_layers(embedded, hidden)
            output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights


class RNN(nn.Module):
    # TODO: make dictionary class and write there
    EOS = "EOS"
    BOS = "BOS"

    def _get_encoder(self):
        raise NotImplementedError

    def _get_decoder(self):
        raise NotImplementedError

    def __init__(self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        src_dictionary,
        tgt_dictionary,
        max_length=64,
        use_attention=False):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.max_length = 64
        self.num_layers = num_layers
        
        # TODO: is this the same dropout as referenced in the paper?
        self.encoder = self._get_encoder()(input_size, hidden_size, num_layers, dropout, src_dictionary)
        self.decoder = self._get_decoder()(hidden_size, num_layers, dropout, tgt_dictionary, use_attention=use_attention)

    def device(self):
        # TODO: consider optimizing
        return next(self.encoder.parameters()).device

    def init_hidden(self):
        # TODO: make it work for bszs
        # TODO: why is it like this?
        if self.num_layers > 1:
            return self.num_layers * [torch.zeros(self.num_layers, 1, self.encoder.hidden_size, device=self.device(), requires_grad=True)]
        return torch.zeros(self.num_layers, 1, self.encoder.hidden_size, device=self.device(), requires_grad=True)

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
            decoder_output, decoder_hidden, _ = self.decoder(
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


# 
#  LSTM variant below
#


class LSTMEncoder(Encoder):
    def _get_hidden_type(self):
        return nn.LSTM


class LSTMDecoder(Decoder):
    def _get_hidden_type(self):
        return nn.LSTM


class LSTMRNN(RNN):
    def _get_encoder(self):
        return LSTMEncoder

    def _get_decoder(self):
        return LSTMDecoder


#
#  GRU variant below
# 


class GRUEncoder(Encoder):
    def _get_hidden_type(self):
        return nn.GRU


class GRUDecoder(Decoder):
    def _get_hidden_type(self):
        return nn.GRU


class GRURNN(RNN):
    def _get_encoder(self):
        return GRUEncoder

    def _get_decoder(self):
        return GRUDecoder



if __name__ == "__main__":
    tasks = "../../data/SCAN/tasks.txt"
    src_dict, tgt_dict = generate_scan_dictionary(tasks, add_bos=True, add_eos=True)
    print("Dictionary loaded")
    dataset = SCANDataset(tasks, src_dict, tgt_dict, device="cpu")
    print("Dataset loaded")
    model = LSTMRNN(
        len(src_dict),
        100,
        2,
        0.5,
        src_dict,
        tgt_dict
    )
    print("LSTM model initialized")
    input, target = dataset[0]
    print(f"Input shape {input.shape}. Target shape {target.shape}.")
    outputs = model.forward(input, target, teacher_forcing=False)
    print("LSTM model forward ran without teacher forcing")
    outputs = model.forward(input, target, teacher_forcing=True)
    print("LSTM model forward ran with teacher forcing")

    print("Starting GRU testing")
    gru_model = GRURNN(
        len(src_dict),
        50,
        1,
        0.5,
        src_dict,
        tgt_dict
    )
    print("GRU Model initialized")
    outputs = gru_model.forward(input, target, teacher_forcing=False)
    print("GRU model forward ran without teacher forcing")
    outputs = gru_model.forward(input, target, teacher_forcing=True)
    print("GRU model forward ran with teacher forcing")
    gru_model = GRURNN(
        len(src_dict),
        50,
        1,
        0.5,
        src_dict,
        tgt_dict,
        use_attention=True
    )
    print("GRU Model with attention initialized")
    outputs = gru_model.forward(input, target, teacher_forcing=False)
    print("GRU model w.att. forward ran without teacher forcing")
    outputs = gru_model.forward(input, target, teacher_forcing=True)
    print("GRU model w.att.forward ran with teacher forcing")
