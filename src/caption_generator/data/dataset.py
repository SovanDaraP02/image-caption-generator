"""ImageCaptionDataset + collate_fn.

Dataset-agnostic: works with any (image_filename, raw_caption) pairs
under a shared image directory, so the same class serves Flickr8k
(see SETUP_DATA.md) and the larger COCO Karpathy-split subset (see
notebooks/train_colab_coco.ipynb) without modification.

collate_fn exists because captions in a batch have different lengths,
and the default DataLoader can't stack tensors of different shapes into
one batch tensor. Each batch is padded to its own longest caption
(not the dataset-wide longest), so short batches don't waste compute.
"""

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from caption_generator.data.vocabulary import Vocabulary

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir: str, image_caption_pairs: list[tuple[str, str]],
                 vocab: Vocabulary, split: str = "train"):
        """
        image_dir: path to the Images/ folder
        image_caption_pairs: list of (image_filename, raw_caption_string)
        vocab: a built Vocabulary instance
        split: "train" (with augmentation) or "val"/"test" (no augmentation)
        """
        self.image_dir = image_dir
        self.pairs = image_caption_pairs
        self.vocab = vocab

        if split == "train":
            self.transform = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, raw_caption = self.pairs[idx]
        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        image = self.transform(image)

        caption_ids = torch.tensor(self.vocab.encode(raw_caption), dtype=torch.long)
        return image, caption_ids


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]],
                pad_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """batch: list of (image, caption_ids) tuples from __getitem__."""
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)  # (B, 3, 224, 224)
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)  # (B, T_max)
    return images, captions
