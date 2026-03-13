import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from src.models.captioning_model import ImageCaptioningModel
from src.training.train_utils import generate_square_subsequent_mask



def generate_caption(model, image_path, vocab, idx_to_word, transform, device, max_len=30):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        # Encode image
        memory = model.encoder(image)  # (1, 49, 512)

        # Start with <start> token
        caption = [vocab["<start>"]]

        for _ in range(max_len):
            caption_tensor = torch.tensor(caption).unsqueeze(0).to(device)  # (1, current_seq_len)

            seq_len = caption_tensor.size(1)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(device)

            outputs = model.decoder(
                caption_tensor,
                memory,
                tgt_mask=tgt_mask
            )

            # Get next token (greedy)
            next_token_logits = outputs[:, -1, :]  # last token prediction
            next_token = next_token_logits.argmax(-1).item()

            caption.append(next_token)

            if next_token == vocab["<end>"]:
                break

    # Convert IDs to words, remove <start> and <end>
    words = [idx_to_word[idx] for idx in caption]
    return " ".join(words[1:-1])


# ── Convenience class ─────────────────────────────────────────────────────────

class CaptionPredictor:


    def __init__(self, model, vocab, image_transform, device, max_len=30):
        self.model = model
        self.vocab = vocab
        self.idx_to_word = {idx: word for word, idx in vocab.items()}
        self.image_transform = image_transform
        self.device = device
        self.max_len = max_len

    @classmethod
    def from_config(cls, cfg, vocab):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # load stats for the transform
        stats = torch.load(cfg.paths.stats_path, map_location="cpu")
        norm_mean = stats["mean"]
        norm_std  = stats["std"]

        
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

        # load model
        model = ImageCaptioningModel(vocab_size=len(vocab), embed_size=cfg.model.embed_size)
        model.load_state_dict(torch.load(cfg.inference.model_path, map_location=device))
        model.to(device)
        model.eval()

        return cls(model, vocab, image_transform, device, max_len=cfg.inference.max_len)

    def predict(self, image_input):
        """Accept a path string, Path, or PIL Image. Returns caption string."""
        if isinstance(image_input, (str, Path)):
            image_path = image_input
        else:
            # it's a PIL Image — save to a temp file path trick OR pass directly
            # We handle PIL Images by patching generate_caption slightly
            image = image_input.convert("RGB")
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            return self._predict_from_tensor(image_tensor)

        return generate_caption(
            self.model,
            image_path,
            self.vocab,
            self.idx_to_word,
            self.image_transform,
            self.device,
            max_len=self.max_len
        )

    def _predict_from_tensor(self, image_tensor):
        """Run the same greedy decode loop on a pre-processed tensor."""
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encoder(image_tensor)
            caption = [self.vocab["<start>"]]

            for _ in range(self.max_len):
                caption_tensor = torch.tensor(caption).unsqueeze(0).to(self.device)
                seq_len = caption_tensor.size(1)
                tgt_mask = generate_square_subsequent_mask(seq_len).to(self.device)
                outputs = self.model.decoder(caption_tensor, memory, tgt_mask=tgt_mask)
                next_token = outputs[:, -1, :].argmax(-1).item()
                caption.append(next_token)
                if next_token == self.vocab["<end>"]:
                    break

        words = [self.idx_to_word[idx] for idx in caption]
        return " ".join(words[1:-1])