
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import torch
from torchvision import transforms

from configs import load_config
from src.models.captioning_model import ImageCaptioningModel
from src.inference.predictor import generate_caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load vocab
    with open(cfg.paths.vocab_path) as f:
        vocab = json.load(f)

    # 
    idx_to_word = {idx: word for word, idx in vocab.items()}

    # load normalisation stats
    stats = torch.load(cfg.paths.stats_path, map_location="cpu")
    norm_mean = stats["mean"]
    norm_std  = stats["std"]

    
    model = ImageCaptioningModel(vocab_size=len(vocab), embed_size=cfg.model.embed_size)
    model.load_state_dict(torch.load(cfg.inference.model_path, map_location=device))
    model.to(device)
    model.eval()

    
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm_mean,
            std=norm_std
        )
    ])

    
    caption = generate_caption(
        model,
        args.image,
        vocab,
        idx_to_word,
        image_transform,
        device
    )

    print("Generated caption:", caption)


if __name__ == "__main__":
    main()