
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class CaptionDataset(Dataset):

    def __init__(self, pairs, vocab, transform=None, max_len=30):
        self.pairs = pairs
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def tokenize(self, caption):
        tokens = caption.lower().split()
        tokens = ["<start>"] + tokens + ["<end>"]

        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]

        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids)

    def __getitem__(self, idx):

        img_path, caption = self.pairs[idx]

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        caption_ids = self.tokenize(caption)

        input_caption = caption_ids[:-1]
        target_caption = caption_ids[1:]

        return image, input_caption, target_caption



def collate_function(batch, mask_token=None):

    image_batch = torch.stack([item[0] for item in batch], dim=0)
    input_embedding_batch = torch.stack([item[1] for item in batch], dim=0)
    output_embedding_batch = torch.stack([item[2] for item in batch], dim=0)
    mask = (input_embedding_batch != mask_token)

    return image_batch, input_embedding_batch, output_embedding_batch, mask



def build_loaders(pairs, vocab, image_transform, batch_size=128, val_ratio=0.1,
                  max_len=30, num_workers=0):
    """
    Builds main_dataLoader, train_loader, val_loader — matching cell 24.
    """
    main_dataset = CaptionDataset(pairs=pairs, vocab=vocab,
                                  transform=image_transform, max_len=max_len)

    val_size = int(len(main_dataset) * val_ratio)
    train_size = len(main_dataset) - val_size
    train_dataset, val_dataset = random_split(main_dataset, [train_size, val_size])

    _collate = lambda batch: collate_function(batch, mask_token=vocab['<pad>'])

    main_dataLoader = DataLoader(
        main_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers
    )

    return main_dataLoader, train_loader, val_loader