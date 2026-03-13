
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x



class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_layers=6, nhead=8,
                 dim_feedforward=2048, max_len=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)  # gonna return the B, Seq_len , d_model=512

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):

        tgt_embed = self.embed(tgt) * (self.embed_size ** 0.5)  # (B, seq_len, embed_size)
        tgt_embed = self.pos_encoder(tgt_embed)  # add positional encoding

        # Transformer expects (seq_len, B, embed_size)
        tgt_embed = tgt_embed.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)

        output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = output.permute(1, 0, 2)  # (B, seq_len, embed_size)
        output = self.fc_out(output)      # (B, seq_len, vocab_size)
        return output