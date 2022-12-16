import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import generate_scan_dictionary, SCANDataset


class Encoder(nn.Module):
    def _get_hidden_type(self):
        raise NotImplementedError

    def __init__(self, input_size, hidden_size, num_layers, dropout, dictionary):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.dictionary = dictionary

        self.embedding = nn.Embedding(input_size, hidden_size)
        layer_type = self._get_hidden_type()
        self.hidden_layers = layer_type(
            hidden_size, hidden_size, num_layers=num_layers, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output = self.dropout(output)
        output, hidden = self.hidden_layers(output, hidden)
        return output, hidden


class Attention(nn.Module):
    """
    Implementation of attention following the description in the
    supplement to the SCAN paper, following Dima's attention paper.
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_hiddens):
        h = self.w(hidden[0]) + self.u(encoder_hiddens)
        mmal = self.v(h.tanh()).squeeze()
        weights = F.softmax(mmal, dim=-1)
        return weights


class Decoder(nn.Module):
    def _get_hidden_type(self):
        raise NotImplementedError

    def __init__(
        self,
        hidden_size,
        num_layers,
        dropout,
        dictionary,
        use_attention=False,
        max_length=49,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.dictionary = dictionary
        self.max_length = max_length
        self.use_attention = use_attention

        self.embedding = nn.Embedding(len(dictionary), hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        layer_type = self._get_hidden_type()
        if use_attention:
            self.attention = Attention(hidden_size=hidden_size)
            self.hidden_layers = layer_type(
                hidden_size, hidden_size, num_layers=num_layers, dropout=dropout
            )
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.out = nn.Linear(hidden_size, len(dictionary))
        else:
            self.hidden_layers = layer_type(
                hidden_size, hidden_size, num_layers=num_layers, dropout=dropout
            )
            self.out = nn.Linear(hidden_size, len(dictionary))

    def forward(self, input, hidden, encoder_hiddens):
        # hidden: [num_layers, bsz, hidden_size]
        # encoder_hiddens: [max_length, hidden_size]
        # input: int
        # embedded: [1,1, hidden_size]
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        attn_weights = None
        if self.use_attention:
            if self.model_type == "lstm":
                attn_weights = self.attention(hidden[0], encoder_hiddens)
            else:
                attn_weights = self.attention(hidden, encoder_hiddens)
            
            context = torch.mm(
                attn_weights.unsqueeze(dim=0),
                encoder_hiddens,
            )
            output = torch.cat((context.squeeze(), embedded.squeeze()), dim=-1)
            output = self.attn_combine(output).unsqueeze(0)
            output = output.view(1, 1, -1)
            # As in torch tutorial
            output = output.relu()

            output, hidden = self.hidden_layers(output, hidden)

            # From the supplement:
            # "Last the hidden state is concatenated with c_i and mapped to a softmax"
            if self.model_type == "lstm":
                #output = torch.cat((context.view(1, 1, -1), hidden[0]), dim=-1)
                output = context.view(1, 1, -1) + hidden[0]
            else:
                #output = torch.cat((context.view(1, 1, -1), hidden), dim=-1)
                output = context.view(1, 1, -1) + hidden

            output = self.out(output)
        else:
            output = embedded
            output, hidden = self.hidden_layers(output, hidden)
            output = self.out(output)
        return output, hidden, attn_weights


class RNN(nn.Module):
    # TODO: make dictionary class and write there
    EOS = "EOS"
    BOS = "BOS"

    model_type = None

    def _get_encoder(self):
        raise NotImplementedError

    def _get_decoder(self):
        raise NotImplementedError

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        src_dictionary,
        tgt_dictionary,
        max_length=64,
        use_attention=False,
    ):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.encoder = self._get_encoder()(
            input_size, hidden_size, num_layers, dropout, src_dictionary
        )
        self.decoder = self._get_decoder()(
            hidden_size,
            num_layers,
            dropout,
            tgt_dictionary,
            use_attention=use_attention,
        )

    def device(self):
        return next(self.encoder.parameters()).device

    def init_hidden(self):
        if self.model_type == "lstm":
            return 2 * [
                torch.zeros(
                    self.num_layers,
                    1,
                    self.encoder.hidden_size,
                    device=self.device(),
                    requires_grad=True,
                )
            ]
        init_weights = torch.zeros(
            self.num_layers,
            1,
            self.encoder.hidden_size,
            device=self.device(),
            requires_grad=True,
        )
        return init_weights

    def forward(self, input, target, teacher_forcing=False, use_oracle=False, evaluate=False):
        input_length = input.shape[0]
        target_length = target.shape[0]

        decoder_max_len = self.max_length
        if not evaluate: #teacher_forcing or use_oracle:
            decoder_max_len = target_length

        # Store state for encoder steps
        encoder_hiddens = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=self.device()
        )

        encoder_hidden = self.init_hidden()
        for idx in range(input_length):
            _enc_output, encoder_hidden = self.encoder(input[idx], encoder_hidden)
            if self.model_type == "lstm":
                (enc_hidden, _enc_cell) = encoder_hidden
            else:
                enc_hidden = encoder_hidden

            # -1 for the last layer! is this right?
            # (This is outputs in the pytorch tutorial)
            # In the paper it is clear that the hidden states are stored,
            # in particular the last layer.
            encoder_hiddens[idx] = enc_hidden[-1]
  
        decoder_input = torch.tensor(
            self.decoder.dictionary[self.BOS], device=self.device()
        )
        decoder_hidden = encoder_hidden

        decoder_outputs = []

        for idx in range(decoder_max_len):
            decoder_output, decoder_hidden, _decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_hiddens
            )

            decoder_outputs.append(decoder_output)

            if teacher_forcing:
                decoder_input = target[idx]  # Teacher forcing
            else:
                _topv, topi = decoder_output.topk(1)
                decoder_input = (
                    topi.squeeze().clone().detach()
                )  # detach from history as input


            # Continue over EOS if not reached length of target
            # if not evaluate:
            # if (use_oracle or teacher_forcing) and len(decoder_outputs) < len(target):
            #    continue

            if decoder_input.item() == self.decoder.dictionary[self.EOS]:
                if not use_oracle:
                    break
                if idx == len(target) - 1:
                    break

                # We force something else than EOS when using oracle
                _topv, topi = decoder_output.topk(2)
                decoder_input = (
                    topi.squeeze()[1].clone().detach()
                )  # detach from history as input
                decoder_outputs[-1][0][0][self.decoder.dictionary[self.EOS]] += -100

        if use_oracle:
            # Make last output EOS
            decoder_outputs[-1][0][0][self.decoder.dictionary[self.EOS]] += 100

        return torch.stack(decoder_outputs)


#
#  LSTM variant below
#


class LSTMEncoder(Encoder):
    model_type = "lstm"

    def _get_hidden_type(self):
        return nn.LSTM


class LSTMDecoder(Decoder):
    model_type = "lstm"

    def _get_hidden_type(self):
        return nn.LSTM


class LSTMRNN(RNN):
    model_type = "lstm"

    def _get_encoder(self):
        return LSTMEncoder

    def _get_decoder(self):
        return LSTMDecoder


#
#  GRU variant below
#


class GRUEncoder(Encoder):
    model_type = "gru"

    def _get_hidden_type(self):
        return nn.GRU


class GRUDecoder(Decoder):
    model_type = "gru"

    def _get_hidden_type(self):
        return nn.GRU


class GRURNN(RNN):
    model_type = "gru"

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
    model = LSTMRNN(len(src_dict), 100, 2, 0.5, src_dict, tgt_dict)
    print("LSTM model initialized")
    input, target = dataset[0]
    print(f"Input shape {input.shape}. Target shape {target.shape}.")
    outputs = model.forward(input, target, teacher_forcing=False)
    print("LSTM model forward ran without teacher forcing")
    outputs = model.forward(input, target, teacher_forcing=True)
    print("LSTM model forward ran with teacher forcing")

    print("Starting GRU testing")
    gru_model = GRURNN(len(src_dict), 50, 1, 0.5, src_dict, tgt_dict)
    print("GRU Model initialized")
    outputs = gru_model.forward(input, target, teacher_forcing=False)
    print("GRU model forward ran without teacher forcing")
    outputs = gru_model.forward(input, target, teacher_forcing=True)
    print("GRU model forward ran with teacher forcing")
    gru_model = GRURNN(
        len(src_dict), 50, 1, 0.5, src_dict, tgt_dict, use_attention=True
    )
    print("GRU Model with attention initialized")
    outputs = gru_model.forward(input, target, teacher_forcing=False)
    print("GRU model w.att. forward ran without teacher forcing")
    outputs = gru_model.forward(input, target, teacher_forcing=True)
    print("GRU model w.att.forward ran with teacher forcing")

    breakpoint()
