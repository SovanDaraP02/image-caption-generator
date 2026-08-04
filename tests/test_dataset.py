import os

from PIL import Image
from torch.utils.data import DataLoader

from vocabulary import Vocabulary
from dataset import Flickr8kDataset, collate_fn


def _make_fake_dataset(tmp_path):
    captions_pool = [
        "a brown dog running across a grassy field",
        "a cat sits on the mat",
        "two people walking on a beach at sunset",
    ]
    image_dir = tmp_path / "images"
    os.makedirs(image_dir, exist_ok=True)

    pairs = []
    for i, cap in enumerate(captions_pool):
        fname = f"fake_{i}.jpg"
        Image.new("RGB", (300, 300), color=(i * 50, 100, 150)).save(image_dir / fname)
        pairs.append((fname, cap))

    vocab = Vocabulary(min_word_freq=1).build(captions_pool)
    return str(image_dir), pairs, vocab


def test_batch_shapes(tmp_path):
    image_dir, pairs, vocab = _make_fake_dataset(tmp_path)
    dataset = Flickr8kDataset(image_dir, pairs, vocab, split="train")
    pad_idx = vocab.word2idx[vocab.PAD_TOKEN]

    loader = DataLoader(dataset, batch_size=3, shuffle=False,
                         collate_fn=lambda b: collate_fn(b, pad_idx))
    images, captions = next(iter(loader))

    assert images.shape == (3, 3, 224, 224)
    assert captions.shape[0] == 3


def test_captions_padded_to_batch_max_length(tmp_path):
    image_dir, pairs, vocab = _make_fake_dataset(tmp_path)
    dataset = Flickr8kDataset(image_dir, pairs, vocab, split="val")
    pad_idx = vocab.word2idx[vocab.PAD_TOKEN]

    loader = DataLoader(dataset, batch_size=len(pairs), shuffle=False,
                         collate_fn=lambda b: collate_fn(b, pad_idx))
    _, captions = next(iter(loader))

    expected_max_len = max(len(vocab.encode(cap)) for _, cap in pairs)
    assert captions.shape[1] == expected_max_len


def test_val_split_has_no_random_augmentation_flip(tmp_path):
    image_dir, pairs, vocab = _make_fake_dataset(tmp_path)
    train_ds = Flickr8kDataset(image_dir, pairs, vocab, split="train")
    val_ds = Flickr8kDataset(image_dir, pairs, vocab, split="val")

    train_transform_names = [type(t).__name__ for t in train_ds.transform.transforms]
    val_transform_names = [type(t).__name__ for t in val_ds.transform.transforms]

    assert "RandomHorizontalFlip" in train_transform_names
    assert "RandomHorizontalFlip" not in val_transform_names
