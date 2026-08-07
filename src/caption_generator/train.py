"""Training loop. Run on a machine with a GPU and real Flickr8k data
downloaded (see SETUP_DATA.md, or notebooks/train_colab.ipynb for a
ready-to-run Colab pipeline).

Gradient clipping guards against the exploding gradients LSTMs are
prone to -- without it, a single bad batch can blow up the weights
into NaN partway through a long run. The doubly stochastic attention
regularization term encourages the model to attend to every image
region roughly equally over the course of a full caption (Xu et al.
2015, Show Attend and Tell, Section 4.2.1), added below as
`attention_regularization`. The checkpoint saved is whichever epoch had
the best validation loss, not the last epoch -- on a dataset this small
those are often not the same epoch.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from caption_generator.data.dataset import ImageCaptionDataset, collate_fn
from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCNN


def train_one_epoch(encoder: EncoderCNN, decoder: DecoderWithAttention,
                     loader: DataLoader, optimizer: torch.optim.Optimizer,
                     criterion: nn.Module, device: torch.device, pad_idx: int,
                     alpha_reg_lambda: float = 1.0, grad_clip: float = 5.0) -> float:
    decoder.train()
    total_loss = 0.0

    for images, captions in loader:
        images, captions = images.to(device), captions.to(device)

        encoder_out = encoder(images)                       # (B, 49, 2048)
        logits, alphas = decoder(encoder_out, captions)      # teacher forcing

        targets = captions[:, 1:]  # predict tokens 1..T-1
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        # doubly stochastic attention regularization: encourage
        # sum-over-time attention per region to be close to 1
        attention_regularization = alpha_reg_lambda * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        loss = loss + attention_regularization

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(encoder: EncoderCNN, decoder: DecoderWithAttention, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> float:
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


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # See SETUP_DATA.md for downloading Flickr8k + captions.txt, or run
    # notebooks/train_colab.ipynb, which builds these from the real
    # dataset and calls train_one_epoch/validate directly.
    TRAIN_PAIRS: list[tuple[str, str]] = []   # (image_filename, raw_caption)
    VAL_PAIRS: list[tuple[str, str]] = []
    IMAGE_DIR = "data/flickr8k/Images"
    TRAIN_CAPTIONS_RAW = [cap for _, cap in TRAIN_PAIRS]

    vocab = Vocabulary(min_word_freq=5).build(TRAIN_CAPTIONS_RAW)
    pad_idx = vocab.word2idx[vocab.PAD_TOKEN]

    train_dataset = ImageCaptionDataset(IMAGE_DIR, TRAIN_PAIRS, vocab, split="train")
    val_dataset = ImageCaptionDataset(IMAGE_DIR, VAL_PAIRS, vocab, split="val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                               collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, pad_idx))

    encoder = EncoderCNN(fine_tune=False).to(device)
    decoder = DecoderWithAttention(vocab_size=len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=4e-4)

    # Halve the LR when val_loss plateaus for 2 epochs -- a fixed LR for
    # the whole run tends to overshoot once the decoder is past its
    # first few epochs of easy gains.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    NUM_EPOCHS = 15  # early stopping below decides the real stopping point
    EARLY_STOP_PATIENCE = 4  # epochs with no val_loss improvement before quitting
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(encoder, decoder, train_loader, optimizer,
                                      criterion, device, pad_idx)
        val_loss = validate(encoder, decoder, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  lr={current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "vocab_word2idx": vocab.word2idx,
                "vocab_idx2word": vocab.idx2word,
            }, "best_checkpoint.pth")
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"No val_loss improvement for {EARLY_STOP_PATIENCE} epochs -- stopping early.")
                break


if __name__ == "__main__":
    main()
