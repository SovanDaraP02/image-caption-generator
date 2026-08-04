"""
Training loop. Run this on Colab with a GPU, real Flickr8k data, and
pretrained ResNet-50 weights (all three need real internet access this
sandbox doesn't have -- see SETUP_DATA.md for the Colab-side steps).

Day 10-11 concept check:
- Gradient clipping: LSTMs are prone to exploding gradients. Without
  clipping, a single bad batch can blow up your weights into NaN and
  you'll have wasted an hour of training with nothing to show for it.
- The doubly stochastic attention regularization term (blueprint
  Section 4.2) encourages the model to look at every image region
  roughly equally *summed over the whole caption* -- it's added to the
  loss below as `attention_regularization`.
- Save the checkpoint with the BEST validation loss, not the last epoch
  -- they're often not the same, especially on a small dataset.
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "data")
sys.path.insert(0, "models")
from vocabulary import Vocabulary          # noqa: E402
from dataset import Flickr8kDataset, collate_fn  # noqa: E402
from encoder import EncoderCNN              # noqa: E402
from decoder import DecoderWithAttention    # noqa: E402


def train_one_epoch(encoder, decoder, loader, optimizer, criterion, device,
                     pad_idx, alpha_reg_lambda=1.0, grad_clip=5.0):
    decoder.train()
    total_loss = 0.0

    for images, captions in loader:
        images, captions = images.to(device), captions.to(device)

        encoder_out = encoder(images)                       # (B, 49, 2048)
        logits, alphas = decoder(encoder_out, captions)      # teacher forcing

        targets = captions[:, 1:]  # predict tokens 1..T-1
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        # doubly stochastic attention regularization (blueprint Section 4.2):
        # encourage sum-over-time attention per region to be close to 1
        attention_regularization = alpha_reg_lambda * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        loss = loss + attention_regularization

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(encoder, decoder, loader, criterion, device):
    decoder.eval()
    total_loss = 0.0
    for images, captions in loader:
        images, captions = images.to(device), captions.to(device)
        encoder_out = encoder(images)
        logits, _ = decoder(encoder_out, captions)
        targets = captions[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- TODO (Colab): point these at your real downloaded data ---
    # See SETUP_DATA.md for how to get Flickr8k + captions.txt onto Colab.
    TRAIN_PAIRS = []   # list of (image_filename, raw_caption) for train split
    VAL_PAIRS = []     # list of (image_filename, raw_caption) for val split
    IMAGE_DIR = "data/flickr8k/Images"
    TRAIN_CAPTIONS_RAW = [cap for _, cap in TRAIN_PAIRS]

    vocab = Vocabulary(min_word_freq=5).build(TRAIN_CAPTIONS_RAW)
    pad_idx = vocab.word2idx[vocab.PAD_TOKEN]

    train_dataset = Flickr8kDataset(IMAGE_DIR, TRAIN_PAIRS, vocab, split="train")
    val_dataset = Flickr8kDataset(IMAGE_DIR, VAL_PAIRS, vocab, split="val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                               collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, pad_idx))

    encoder = EncoderCNN(fine_tune=False).to(device)
    decoder = DecoderWithAttention(vocab_size=len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=4e-4)

    NUM_EPOCHS = 8  # 5-8 is enough in a 15-day timeline; don't chase 10+
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(encoder, decoder, train_loader, optimizer,
                                      criterion, device, pad_idx)
        val_loss = validate(encoder, decoder, val_loader, criterion, device)
        print(f"Epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "vocab_word2idx": vocab.word2idx,
                "vocab_idx2word": vocab.idx2word,
            }, "best_checkpoint.pth")
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
