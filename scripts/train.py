import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os

import torch
import torch.nn as nn

from configs import load_config
from src.data.prepare import prepare_data
from src.data.dataset import build_loaders
from src.models.captioning_model import ImageCaptioningModel
from src.training.train_utils import train_one_epoch, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--resume",  default=None,
                        help="Path to a saved .pth to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if torch.cuda.is_available():
        device = "cuda"
        datatype = torch.float16
    else:
        device = "cpu"
        datatype = torch.float32
    print(f"The device is : {device} \nThe datatype is :{datatype}")

    # ── data 
    data = prepare_data(cfg)
    pairs          = data["pairs"]
    vocab          = data["vocab"]
    image_transform = data["image_transform"]

    main_dataLoader, train_loader, val_loader = build_loaders(
        pairs=pairs,
        vocab=vocab,
        image_transform=image_transform,
        batch_size=cfg.training.batch_size,
        val_ratio=cfg.dataset.val_ratio,
        max_len=cfg.dataset.max_caption_len,
        num_workers=cfg.training.num_workers,
    )
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_size=cfg.model.embed_size
    )
    model.load_state_dict(torch.load(cfg.inference.model_path, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"], label_smoothing=0.1)

    # Freeze all CNN parameters first
    for param in model.encoder.cnn.parameters():
        param.requires_grad = False

    # Unfreeze the last block (layer7)
    for name, param in model.encoder.cnn.named_parameters():
        if name.startswith("cnn.7"):   # cnn.7 corresponds to last Bottleneck block in your CNN
            param.requires_grad = True

    # Unfreeze the fc layer
    for param in model.encoder.fc.parameters():
        param.requires_grad = True

    # Your existing optimizer
    optimizer = torch.optim.Adam([
        {"params": model.encoder.fc.parameters(), "lr": 3e-4},
        {"params": [p for n, p in model.encoder.cnn.named_parameters() if n.startswith("cnn.7")], "lr": 1e-4},
        {"params": model.decoder.parameters(), "lr": 3e-4}
    ])

    # Add the scheduler here
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",    # We want the loss to go DOWN (minimized)
        factor=0.5,    # New LR = Old LR * 0.5
        patience=2,    # Wait 2 epochs before dropping
        verbose=True   # This will print "Epoch X: reducing learning rate..." in your console
    )

    # ── optional resume 
    save_dir  = cfg.paths.saved_models_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "image_caption_model_final.pth")

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # ── training loop ───────────────────────────────────────────────
    num_epochs = cfg.training.num_epochs

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # save every epoch (keeps last + best)
        epoch_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), epoch_path)

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining finished. Model saved to: {save_path}")


if __name__ == "__main__":
    main()