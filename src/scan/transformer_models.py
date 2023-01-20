import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, dropout_p: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)
        )
        pe = torch.zeros(max_len, 1, dim_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
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
        self.positional_encoder = PositionalEncoding(
            dim_model=n_dim, dropout_p=kwargs["dropout"], max_len=self.max_length
        )
        self.transformer = nn.Transformer(*args, **kwargs)
        self.out = nn.Linear(n_dim, tgt_size)
        self._reset_parameters()

    def forward(self, src, tgt, tgt_mask=None):
        src_output = self.embedding(src)
        tgt_output = self.output_embedding(tgt)

        src_output *= math.sqrt(self.n_dim)
        tgt_output *= math.sqrt(self.n_dim)

        src_output = self.positional_encoder(src_output)
        tgt_output = self.positional_encoder(tgt_output)

        src_output = src_output.permute(1, 0, 2)
        tgt_output = tgt_output.permute(1, 0, 2)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.get_masks(
            src, tgt
        )

        output = self.transformer(
            src_output,
            tgt_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        output = self.out(output)
        return output

    def get_masks(self, src, tgt):
        DEVICE = self.device()
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(
            self.device()
        )
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(
            torch.bool
        )
        # This should be read from the dicts.
        src_padding_mask = src == 13
        tgt_padding_mask = tgt == 6

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def teacher_force_predict(self, src, tgt, tgt_mask=None):
        output = self.forward(src, tgt, tgt_mask=tgt_mask)
        pred = output.reshape(-1, output.shape[-1]).argmax(dim=-1)
        return pred

    def encode(self, src):
        src_output = self.embedding(src)
        src_output *= math.sqrt(self.n_dim)
        src_output = self.positional_encoder(src_output)
        src_output = src_output.permute(1, 0, 2)
        output = self.transformer.encoder(src_output)
        return output

    def decode(self, tgt, memory, tgt_mask=None):
        tgt_output = self.output_embedding(tgt)
        tgt_output *= math.sqrt(self.n_dim)
        tgt_output = self.positional_encoder(tgt_output)
        tgt_output = tgt_output.permute(1, 0, 2)

        if tgt_mask is None:
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                tgt.shape[1]
            ).to(self.device())
        output = self.transformer.decoder(tgt_output, memory, tgt_mask=tgt_mask)
        return output

    def device(self):
        return next(self.transformer.encoder.parameters()).device

    def predict(self, src, tgt, bos, eos, use_oracle=True):
        device = self.device()
        output = torch.ones(1, self.max_length).long().to(device) * bos
        output_probs = []

        for t in range(1, self.max_length):
            tgt_mask = self.transformer.generate_square_subsequent_mask(t).to(device)
            decoder_output = self.forward(src.unsqueeze(dim=0), output[:, :t])

            pred_proba_t = decoder_output[-1, :, :]
            output_prob, output_t = pred_proba_t.data.topk(1)

            output[:, t] = output_t.squeeze()
            output_probs.append(pred_proba_t)

            if t >= len(tgt) + 1:
                break

            if output_t.squeeze() == eos:
                break
        return torch.stack(output_probs)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
