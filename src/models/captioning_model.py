import torch.nn as nn

from src.models.encoder import CNNEncoder
from src.models.decoder import TransformerDecoder

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__()
        self.encoder = CNNEncoder(embed_size=embed_size)
        self.decoder = TransformerDecoder(vocab_size=vocab_size, embed_size=embed_size)

    def forward(self, images, captions, tgt_mask=None, tgt_key_padding_mask=None):
        features = self.encoder(images)  # (B, 49, 512)
        outputs = self.decoder(
            captions,
            memory=features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return outputs