import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

def compute_image_mean_std(loader):
    mean = 0.
    std = 0.
    nb_samples = 0

    for images in tqdm(loader, desc="Computing mean/std"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


def build_vocab(pairs, freq_threshold=1):
    counter = Counter()

    for _, caption in pairs:
        tokens = caption.lower().split()  # simple tokenization
        counter.update(tokens)

    vocab = {}

    # Add special tokens first
    vocab["<pad>"] = 0
    vocab["<start>"] = 1
    vocab["<end>"] = 2
    vocab["<unk>"] = 3

    idx = 4
    for word, count in counter.items():
        if count >= freq_threshold:
            vocab[word] = idx
            idx += 1

    return vocab


def load_coco_pairs(images_dir, annotation_file):
    """Build (image_path, caption) pairs from a COCO annotation file."""
    images_dir = Path(images_dir)
    annotation_file = Path(annotation_file)

    with open(annotation_file, "r") as f:
        data = json.load(f)

    # 
    image_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}

    # 
    image_to_captions = defaultdict(list)
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        file_name = image_id_to_name[image_id]
        image_to_captions[file_name].append(caption)

    # 
    imgpath_to_caption = {}
    for img_name, captions in image_to_captions.items():
        img_path = images_dir / img_name
        imgpath_to_caption[img_path] = captions

    # 
    pairs = []
    for img_path, captions in imgpath_to_caption.items():
        for cap in captions:
            pairs.append((img_path, cap))

    return pairs


class _ImageOnlyDataset(Dataset):
    """Minimal dataset used only for computing mean/std (cell 15)."""
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, _ = self.pairs[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image


def prepare_data(cfg):
    """
    Full pipeline from config. Returns dict with:
        pairs, vocab, image_transform, norm_mean, norm_std
    """
    images_dir      = cfg.paths.images_dir
    annotation_file = cfg.paths.annotation_file
    vocab_path      = Path(cfg.paths.vocab_path)
    stats_path      = Path(cfg.paths.stats_path)
    freq_threshold  = cfg.dataset.freq_threshold
    num_workers     = cfg.training.num_workers

    # ── pairs ─────────────────────────────────────────────────────────────────
    print("Loading COCO pairs...")
    pairs = load_coco_pairs(images_dir, annotation_file)
    print(f"Total (image, caption) pairs: {len(pairs)}")

    # ── mean / std ────────────────────────────────────────────────────────────
    if stats_path.exists():
        print(f"Loading cached stats from {stats_path}")
        stats = torch.load(stats_path)
        norm_mean, norm_std = stats["mean"], stats["std"]
    else:
        #
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = _ImageOnlyDataset(pairs, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers)

        #
        mean, std = compute_image_mean_std(loader=loader)
        print(f"First pass mean/std:  {mean}  {std}")

        #
        norm_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        norm_dataset = _ImageOnlyDataset(pairs, transform=norm_transform)
        norm_loader = DataLoader(norm_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
        norm_mean, norm_std = compute_image_mean_std(loader=norm_loader)
        print(f"Normalised mean/std:  {norm_mean}  {norm_std}")

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": norm_mean, "std": norm_std}, stats_path)
        print(f"Stats saved to {stats_path}")

    # ── vocab ─────────────────────────────────────────────────────────────────
    if vocab_path.exists():
        print(f"Loading cached vocab from {vocab_path}")
        import json as _json
        with open(vocab_path) as f:
            vocab = _json.load(f)
    else:
        # 
        vocab = build_vocab(pairs=pairs, freq_threshold=freq_threshold)
        print(f"Vocab size: {len(vocab)}")
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(vocab_path, "w") as f:
            _json.dump(vocab, f, indent=2)
        print(f"Vocab saved to {vocab_path}")

    # ── final image transform ───────────────────────────────────────
    image_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    return {
        "pairs":           pairs,
        "vocab":           vocab,
        "image_transform": image_transform,
        "norm_mean":       norm_mean,
        "norm_std":        norm_std,
    }