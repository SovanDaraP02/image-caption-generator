"""Local COCO training, adapted from notebooks/train_kaggle_coco.ipynb to run
directly on this machine instead of in a Kaggle session -- same data source
(Karpathy split via yerevann/coco-karpathy), same training loop
(caption_generator.train.train_one_epoch/validate), same resumable
checkpoint format, so results are directly comparable to the numbers
already in README.md's Results table.

Run with:
    python scripts/train_local_coco.py                    # full 50k/3k/3k run
    python scripts/train_local_coco.py --n-train 5000      # smaller/faster run

Safe to interrupt (Ctrl-C) and re-run -- it resumes image downloads (skips
files that already exist) and training (from data/coco/latest_checkpoint_coco.pth)
instead of starting over.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

from caption_generator.data.dataset import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, ImageCaptionDataset, collate_fn
from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCLIP, EncoderCNN
from caption_generator.train import train_one_epoch, validate

DATA_DIR = "data/coco"
IMAGE_DIR = os.path.join(DATA_DIR, "Images")
PAIRS_CACHE = os.path.join(DATA_DIR, "pairs_cache.json")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_split_examples(hf_splits: list[str], n: int) -> list[dict]:
    examples = []
    for hf_split in hf_splits:
        if len(examples) >= n:
            break
        stream = load_dataset("yerevann/coco-karpathy", split=hf_split, streaming=True)
        for ex in stream:
            examples.append(ex)
            if len(examples) >= n:
                break
    return examples


def download_images(examples: list[dict], max_workers: int = 32,
                     max_retries: int = 2) -> list[tuple[str, str]]:
    """Downloads to IMAGE_DIR, returns (image_filename, caption) pairs --
    one pair per caption. Skips files that already exist, so re-running
    after an interruption only downloads what's missing."""

    def fetch(ex: dict) -> tuple[dict, bool]:
        path = os.path.join(IMAGE_DIR, ex["filename"])
        if os.path.exists(path):
            return ex, True
        for attempt in range(max_retries + 1):
            try:
                r = requests.get(ex["url"], timeout=10)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                return ex, True
            except Exception:
                if attempt == max_retries:
                    return ex, False
                time.sleep(0.5)
        return ex, False

    pairs = []
    failed = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch, ex) for ex in examples]
        for i, future in enumerate(as_completed(futures), 1):
            ex, ok = future.result()
            if ok:
                for cap in ex["sentences"]:
                    pairs.append((ex["filename"], cap))
            else:
                failed += 1
            if i % 500 == 0 or i == len(examples):
                elapsed = time.time() - start
                print(f"  downloaded {i}/{len(examples)} images "
                      f"({failed} failed) -- {elapsed:.0f}s elapsed", flush=True)
    return pairs


def build_pairs(n_train: int, n_val: int, n_test: int) -> dict[str, list]:
    if os.path.exists(PAIRS_CACHE):
        print(f"Found cached pairs at {PAIRS_CACHE} -- reusing (delete it to re-download).")
        with open(PAIRS_CACHE) as f:
            return json.load(f)

    os.makedirs(IMAGE_DIR, exist_ok=True)

    print(f"Collecting {n_train} train examples (train + restval splits)...")
    train_examples = collect_split_examples(["train", "restval"], n_train)
    print(f"Collecting {n_val} val examples...")
    val_examples = collect_split_examples(["validation"], n_val)
    print(f"Collecting {n_test} test examples...")
    test_examples = collect_split_examples(["test"], n_test)

    print(f"Downloading {len(train_examples)} train images...")
    train_pairs = download_images(train_examples)
    print(f"Downloading {len(val_examples)} val images...")
    val_pairs = download_images(val_examples)
    print(f"Downloading {len(test_examples)} test images...")
    test_pairs = download_images(test_examples)

    data = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    with open(PAIRS_CACHE, "w") as f:
        json.dump(data, f)
    print(f"Cached pairs to {PAIRS_CACHE}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=3_000)
    parser.add_argument("--n-test", type=int, default=3_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder", choices=["resnet50", "clip-vit-base-patch32"], default="resnet50",
                         help="resnet50: ImageNet-classification features (original). "
                              "clip-vit-base-patch32: CLIP's contrastively-pretrained, "
                              "language-aligned features -- see EncoderCLIP's docstring.")
    parser.add_argument("--lr", type=float, default=4e-4, help="decoder learning rate")
    parser.add_argument("--finetune-from", type=str, default=None,
                         help="Warm-start encoder+decoder from this checkpoint instead of training "
                              "from scratch, and unfreeze the encoder's last block "
                              "(EncoderCLIP/EncoderCNN fine_tune=True) with its own, much lower "
                              "learning rate (--encoder-lr). For continuing an already-converged "
                              "frozen-encoder run into a fine-tuning phase, not for resuming an "
                              "interrupted run of this same script (that's automatic, see latest_path).")
    parser.add_argument("--encoder-lr", type=float, default=1e-5,
                         help="learning rate for the newly-unfrozen encoder block when "
                              "--finetune-from is set -- deliberately much lower than --lr so "
                              "large early gradients (decoder is far more converged than the "
                              "encoder is used to seeing) don't destroy the pretrained features.")
    args = parser.parse_args()

    # Suffix checkpoint paths by encoder (and fine-tuning phase) so runs
    # can't clobber each other -- they're different models, meant to be
    # compared, not overwritten.
    suffix = "" if args.encoder == "resnet50" else "_clip"
    if args.finetune_from:
        suffix += "_finetuned"
    latest_path = os.path.join(DATA_DIR, f"latest_checkpoint_coco{suffix}.pth")
    best_path = f"best_checkpoint_coco{suffix}.pth"

    os.makedirs(DATA_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}, encoder: {args.encoder}"
          + (f", fine-tuning from: {args.finetune_from}" if args.finetune_from else ""))

    data = build_pairs(args.n_train, args.n_val, args.n_test)
    train_pairs = [tuple(p) for p in data["train"]]
    val_pairs = [tuple(p) for p in data["val"]]
    test_pairs = [tuple(p) for p in data["test"]]
    print(f"Pairs -- train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}")

    vocab = Vocabulary()
    if args.finetune_from:
        # Reuse the exact vocab the starting checkpoint's decoder was built
        # with -- rebuilding from data (even the same data) risks any
        # nondeterminism producing a subtly different vocab, which would
        # silently corrupt the loaded embedding/output layer alignment.
        finetune_ckpt = torch.load(args.finetune_from, map_location=device)
        vocab.word2idx = finetune_ckpt["vocab_word2idx"]
        vocab.idx2word = finetune_ckpt["vocab_idx2word"]
    else:
        train_captions_raw = [cap for _, cap in train_pairs]
        vocab = Vocabulary(min_word_freq=5).build(train_captions_raw)
    pad_idx = vocab.word2idx[vocab.PAD_TOKEN]
    print(f"Vocab size: {len(vocab)}")

    if args.encoder == "clip-vit-base-patch32":
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    train_dataset = ImageCaptionDataset(IMAGE_DIR, train_pairs, vocab, split="train", mean=mean, std=std)
    val_dataset = ImageCaptionDataset(IMAGE_DIR, val_pairs, vocab, split="val", mean=mean, std=std)

    # num_workers=0: Python 3.14 switched multiprocessing's default start
    # method away from fork on this platform, which can't pickle the
    # closure-based collate_fn lambda below. The GPU (MPS) forward/backward
    # pass is the actual bottleneck here, not single-process data loading.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=lambda b: collate_fn(b, pad_idx), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, pad_idx), num_workers=0)

    fine_tune_encoder = bool(args.finetune_from)
    if args.encoder == "clip-vit-base-patch32":
        encoder = EncoderCLIP(fine_tune=fine_tune_encoder).to(device)
        decoder = DecoderWithAttention(vocab_size=len(vocab), encoder_dim=EncoderCLIP.OUTPUT_DIM).to(device)
    else:
        encoder = EncoderCNN(fine_tune=fine_tune_encoder).to(device)
        decoder = DecoderWithAttention(vocab_size=len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1

    if args.finetune_from:
        print(f"Warm-starting encoder+decoder from {args.finetune_from}")
        encoder.load_state_dict(finetune_ckpt["encoder_state"])
        decoder.load_state_dict(finetune_ckpt["decoder_state"])

        # Differential learning rates: the decoder is already well-converged
        # on frozen features (--lr, same as initial training), but the
        # newly-unfrozen encoder block has never been updated by gradient
        # descent for this task at all -- a shared high LR would blow away
        # its pretrained weights in a step or two. encoder_params is
        # whichever ones fine_tune=True actually unfroze (requires_grad=True).
        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": decoder.parameters(), "lr": args.lr},
            {"params": encoder_params, "lr": args.encoder_lr},
        ])

        # Establish this run's real baseline before training -- comparing
        # against the frozen-encoder starting point, not an arbitrary
        # infinity, so a checkpoint only gets saved as "best" if fine-tuning
        # the encoder actually beat where it started.
        best_val_loss = validate(encoder, decoder, val_loader, criterion, device)
        print(f"Starting val_loss (frozen-encoder baseline, before fine-tuning): {best_val_loss:.4f}")
    else:
        optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        checkpoint_vocab_size = len(ckpt["vocab_word2idx"])
        checkpoint_encoder = ckpt.get("encoder_type", "resnet50")
        # Vocab is derived from this run's specific training corpus (--n-train
        # etc.), and encoder_dim depends on which encoder produced the
        # checkpoint -- either mismatch means incompatible layer shapes, so
        # resuming would corrupt training rather than continue it. Safer to
        # start fresh and warn loudly than to silently resume broken or crash
        # mid-run.
        if checkpoint_vocab_size != len(vocab):
            print(f"WARNING: found {latest_path}, but its vocab size ({checkpoint_vocab_size}) doesn't "
                  f"match this run's vocab ({len(vocab)}) -- it's from a different-sized run. "
                  "Starting fresh instead of resuming. Move or delete that file to silence this warning.")
        elif checkpoint_encoder != args.encoder:
            print(f"WARNING: found {latest_path}, but it was trained with encoder '{checkpoint_encoder}', "
                  f"not this run's '{args.encoder}'. Starting fresh instead of resuming. "
                  "Move or delete that file to silence this warning.")
        else:
            print(f"Found a previous run's checkpoint at {latest_path} -- resuming instead of starting over.")
            encoder.load_state_dict(ckpt["encoder_state"])
            decoder.load_state_dict(ckpt["decoder_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            vocab.word2idx = ckpt["vocab_word2idx"]
            vocab.idx2word = ckpt["vocab_idx2word"]
            pad_idx = vocab.word2idx[vocab.PAD_TOKEN]
            best_val_loss = ckpt["best_val_loss"]
            epochs_without_improvement = ckpt["epochs_without_improvement"]
            start_epoch = ckpt["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}, best_val_loss so far = {best_val_loss:.4f}")
    else:
        print("No previous checkpoint found -- starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(encoder, decoder, train_loader, optimizer, criterion, device, pad_idx)
        val_loss = validate(encoder, decoder, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={current_lr:.2e}  ({elapsed:.0f}s)", flush=True)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save({
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "vocab_word2idx": vocab.word2idx,
            "vocab_idx2word": vocab.idx2word,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "encoder_type": args.encoder,
        }, latest_path)

        if improved:
            torch.save({
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "vocab_word2idx": vocab.word2idx,
                "vocab_idx2word": vocab.idx2word,
                "encoder_type": args.encoder,
            }, best_path)
            print(f"  -> saved new best checkpoint to {best_path} (val_loss={val_loss:.4f})")

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"No val_loss improvement for {args.early_stop_patience} epochs -- stopping early.")
            break

    print("\nTraining done. Evaluating on test split...")
    test_pairs_by_image = defaultdict(list)
    for fname, cap in test_pairs:
        test_pairs_by_image[fname].append(cap)

    # skip_meteor=True: pycocoevalcap's METEOR scorer shells out to a Java
    # subprocess that has been observed in this project to hang cleanup
    # indefinitely on failure (see README's Engineering notes / evaluate.py).
    # BLEU and CIDEr don't share that dependency.
    from caption_generator.evaluate import evaluate
    scores = evaluate(best_path, dict(test_pairs_by_image), IMAGE_DIR, device=device, skip_meteor=True)
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
