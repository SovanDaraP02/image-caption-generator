"""
Flickr8kDataset + collate_fn.

Day 4 concept check: why a custom collate_fn?
Captions in a batch have different lengths. The default DataLoader can't
stack tensors of different shapes into one batch tensor, so we pad every
caption in the batch up to the longest one *in that batch* (not the
longest in the whole dataset -- that would waste compute on tiny batches).

Expected Flickr8k layout after download (see ../SETUP_DATA.md):
    data/flickr8k/
        Images/
            1000268201_693b08cb0e.jpg
            ...
        captions.txt        # columns: image,caption
"""

import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms as T


class Flickr8kDataset(Dataset):
    def __init__(self, image_dir: str, image_caption_pairs: list[tuple[str, str]],
                 vocab, split: str = "train"):
        """
        image_dir: path to the Images/ folder
        image_caption_pairs: list of (image_filename, raw_caption_string)
        vocab: a built Vocabulary instance
        split: "train" (with augmentation) or "val"/"test" (no augmentation)
        """
        self.image_dir = image_dir
        self.pairs = image_caption_pairs
        self.vocab = vocab

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if split == "train":
            self.transform = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(imagenet_mean, imagenet_std),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(imagenet_mean, imagenet_std),
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, raw_caption = self.pairs[idx]
        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        image = self.transform(image)

        caption_ids = torch.tensor(self.vocab.encode(raw_caption), dtype=torch.long)
        return image, caption_ids


def collate_fn(batch, pad_idx: int):
    """batch: list of (image, caption_ids) tuples from __getitem__."""
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)  # (B, 3, 224, 224)
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)  # (B, T_max)
    return images, captions


if __name__ == "__main__":
    # logic self-test using synthetic in-memory images (no real Flickr8k
    # download needed to verify this part of the pipeline works) —
    # run with: python dataset.py
    import sys
    sys.path.insert(0, ".")
    from vocabulary import Vocabulary

    os.makedirs("/tmp/fake_images", exist_ok=True)
    fake_pairs = []
    captions_pool = [
        "a brown dog running across a grassy field",
        "a cat sits on the mat",
        "two people walking on a beach at sunset",
    ]
    for i, cap in enumerate(captions_pool):
        fname = f"fake_{i}.jpg"
        Image.new("RGB", (300, 300), color=(i * 50, 100, 150)).save(f"/tmp/fake_images/{fname}")
        fake_pairs.append((fname, cap))

    vocab = Vocabulary(min_word_freq=1).build(captions_pool)
    dataset = Flickr8kDataset("/tmp/fake_images", fake_pairs, vocab, split="train")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=3, shuffle=True,
                         collate_fn=lambda b: collate_fn(b, vocab.word2idx[vocab.PAD_TOKEN]))

    images, captions = next(iter(loader))
    print(f"images batch shape: {tuple(images.shape)}  (expected: (3, 3, 224, 224))")
    print(f"captions batch shape: {tuple(captions.shape)}  (expected: (3, T_max))")
    assert images.shape == (3, 3, 224, 224)
    assert captions.shape[0] == 3

    print("dataset.py self-test passed")
