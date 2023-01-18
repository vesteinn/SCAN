import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, dim_model: int, dropout_p: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
        pe = torch.zeros(max_len, 1, dim_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SCANTransformer(nn.Module):

    model_type = "transformer"
    max_length = 50

    def __init__(self, *args, **kwargs):
        super().__init__()
        n_dim = kwargs["d_model"]
        self.n_dim = n_dim
        src_size = kwargs["src_size"]
        tgt_size = kwargs["tgt_size"]
        del kwargs["src_size"]
        del kwargs["tgt_size"]
        self.args = args
        self.kwargs = kwargs

        self.embedding = nn.Embedding(src_size, n_dim)
        self.output_embedding = nn.Embedding(tgt_size, n_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.positional_encoder = PositionalEncoding(
            dim_model=n_dim, dropout_p=kwargs["dropout"], max_len=self.max_length
        )
        self.positional_encoder_tgt = PositionalEncoding(
            dim_model=n_dim, dropout_p=kwargs["dropout"], max_len=self.max_length
        )
        self.transformer = nn.Transformer(
            *args, **kwargs
        )
        self.out = nn.Linear(n_dim, tgt_size)
        self._reset_parameters()

    def forward(self, src, tgt, tgt_mask=None, src_mask=None, training=True):

        src_output = self.dropout(self.embedding(src))
        tgt_output = self.dropout(self.output_embedding(tgt))
        
        src_output *= math.sqrt(self.n_dim)
        tgt_output *= math.sqrt(self.n_dim)
        
        src_output = self.positional_encoder(src_output)
        tgt_output = self.positional_encoder(tgt_output)

        src_output = src_output.permute(1, 0, 2)
        tgt_output = tgt_output.permute(1, 0, 2)
        
        if tgt_mask is None:
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device())

        output = self.transformer(src_output, tgt_output, tgt_mask=tgt_mask, src_mask=src_mask)
        output = self.out(output)
        return output

    def device(self):
        return next(self.transformer.encoder.parameters()).device

    def predict(self, src, tgt, bos, eos, use_oracle=True):
        #src_output = self.embedding(src)
        #encoder_output = self.transformer.encoder(src_output).unsqueeze(dim=1)
        
        device = self.device()

        output = torch.ones(1, self.max_length).long().to(device) * bos
        
        output_probs = []

        for t in range(1, self.max_length):
            # Shift by one to predict last
            #tgt_emb = self.output_embedding(output[:, :t]).transpose(1, 0)
            
            tgt_mask = self.transformer.generate_square_subsequent_mask(t).to(device)
            
            #breakpoint()
            #decoder_output = self.transformer.decoder(
            #    tgt=tgt_emb,
            #    memory=encoder_output,
            #    tgt_mask=tgt_mask
            #)

            decoder_output = self.forward(src.unsqueeze(dim=0), output[:, :t], tgt_mask=tgt_mask, training=False)            
            #decoder_output = decoder_output.permute(1, 2, 0)

            #breakpoint()
            pred_proba_t = decoder_output[-1, :, :]
            output_prob, output_t = pred_proba_t.data.topk(1)

            #breakpoint() 
            output[:, t] = output_t.squeeze()
            output_probs.append(pred_proba_t)

            #breakpoint()

            if t >= len(tgt) + 2:
                return torch.stack(output_probs)

            if output_t.squeeze() == eos:
                return torch.stack(output_probs)

        return torch.stack(output_probs)

    def predictTry(self, src, tgt, bos, eos, use_oracle=True):
        model = self
        input_sequence = src
        SOS_token = bos
        EOS_token = 3
        max_length = self.max_length
        device = self.device()

        model.eval()
        
        y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

        num_tokens = len(input_sequence)

        for _ in range(max_length):
            # Get source mask
            tgt_mask = self.transformer.generate_square_subsequent_mask(y_input.size(1)).to(device)
            
            pred = model(input_sequence, y_input.unsqueeze(dim=0), tgt_mask=tgt_mask)
            
            next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            next_item = torch.tensor([[next_item]], device=device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_token:
                break

        return y_input.view(-1).tolist()


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)