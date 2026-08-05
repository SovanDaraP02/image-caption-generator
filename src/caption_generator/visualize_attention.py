"""Generates attention heatmap overlays: for each generated word, shows
which of the 49 image regions the model attended to. This reproduces
the headline visual from Xu et al.'s "Show, Attend and Tell".

Usage (after training):
    python -m caption_generator.visualize_attention \\
        --checkpoint best_checkpoint.pth --image path/to/image.jpg
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import zoom

from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.caption_model import CaptionModel
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCNN


def visualize(checkpoint_path: str, image_path: str, out_path: str = "attention_heatmap.png") -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    vocab = Vocabulary()
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = checkpoint["vocab_idx2word"]

    encoder = EncoderCNN(fine_tune=False)
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder = DecoderWithAttention(vocab_size=len(vocab))
    decoder.load_state_dict(checkpoint["decoder_state"])

    model = CaptionModel(encoder, decoder, vocab, device="cpu")

    raw_image = Image.open(image_path).convert("RGB").resize((224, 224))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(raw_image).unsqueeze(0)

    caption, alphas = model.generate_greedy(image_tensor)
    words = caption.split()

    n_words = min(len(words), len(alphas))
    cols = min(5, n_words)
    rows = (n_words + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(n_words):
        ax = axes[i]
        ax.imshow(raw_image)

        alpha_map = alphas[i].reshape(7, 7).numpy()
        alpha_map = zoom(alpha_map, 224 / 7, order=1)  # upsample 7x7 -> 224x224
        ax.imshow(alpha_map, alpha=0.6, cmap="jet")
        ax.set_title(words[i], fontsize=12)
        ax.axis("off")

    for i in range(n_words, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    print(f"Generated caption: {caption}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="attention_heatmap.png")
    args = parser.parse_args()
    visualize(args.checkpoint, args.image, args.out)
