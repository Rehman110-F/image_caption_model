
import torch
import torch.nn as nn
from tqdm import tqdm



def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask




def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Training")

    for images, input_captions, target_captions, pad_mask in loop:
        images = images.to(device)
        input_captions = input_captions.to(device)
        target_captions = target_captions.to(device)
        pad_mask = pad_mask.to(device)

        seq_len = input_captions.size(1)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(device)

        outputs = model(
            images,
            input_captions,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~pad_mask
        )

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            target_captions.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, input_captions, target_captions, pad_mask in val_loader:
            images = images.to(device)
            input_captions = input_captions.to(device)
            target_captions = target_captions.to(device)
            pad_mask = pad_mask.to(device)

            seq_len = input_captions.size(1)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(device)

            outputs = model(
                images,
                input_captions,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=~pad_mask
            )

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                target_captions.reshape(-1)
            )

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    return avg_loss