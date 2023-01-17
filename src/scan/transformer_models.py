import torch
from torch import nn


class SCANTransformer(nn.Module):

    model_type = "transformer"
    max_length = 50

    def __init__(self, *args, **kwargs):
        super().__init__()
        n_dim = kwargs["d_model"]
        src_size = kwargs["src_size"]
        tgt_size = kwargs["tgt_size"]
        del kwargs["src_size"]
        del kwargs["tgt_size"]
        self.args = args
        self.kwargs = kwargs

        self.embedding = nn.Embedding(src_size, n_dim)
        self.output_embedding = nn.Embedding(tgt_size, n_dim)

        self.transformer = nn.Transformer(
            *args, **kwargs
        )
        self.out = nn.Linear(n_dim, tgt_size)

    def forward(self, src, tgt):
        src_output = self.embedding(src) #.view(1, 1, -1)
        tgt_output = self.output_embedding(tgt) #.view(1, 1, -1)
        output = self.transformer(src_output, tgt_output)
        output = self.out(output)
        return output

    def device(self):
        return next(self.transformer.encoder.parameters()).device

    def predict(self, src, tgt, bos, eos, use_oracle=True):
        src_output = self.embedding(src)
        encoder_output = self.transformer.encoder(src_output).unsqueeze(dim=1)
        
        device = self.device()

        output = torch.ones(1, self.max_length).long().to(device) * bos
        output_probs = []

        for t in range(1, self.max_length):
            tgt_emb = self.output_embedding(output[:, :t]).transpose(0, 1)
            tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(
                t).to(device).transpose(0, 1)
            decoder_output = self.transformer.decoder(
                tgt=tgt_emb,
                memory=encoder_output,
                tgt_mask=tgt_mask
            )

            pred_proba_t = self.out(decoder_output)[-1, :, :]
            output_prob, output_t = pred_proba_t.data.topk(1)
            
            output[:, t] = output_t.squeeze()
            output_probs.append(pred_proba_t)

            if output_t.squeeze() == eos:
                # Cutoff BOS and EOS
                return torch.stack(output_probs) #[:, 1:-1]

        # Cutoff the BOS
        return torch.stack(output_probs) #[:, 1:]