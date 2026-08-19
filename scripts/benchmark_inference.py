"""Measures real inference latency for a trained checkpoint: encoder
forward pass, greedy decoding, and beam-search decoding, on whichever
devices are available (CPU always; MPS/CUDA if present).

Not a training benchmark -- this answers a different, deployment-facing
question: "if this were serving requests, how many milliseconds does one
image cost?" That's what determines hosting cost and whether a synchronous
HTTP endpoint (see api.py) is viable at all versus needing a queue.

Usage:
    python scripts/benchmark_inference.py --checkpoint best_checkpoint.pth --n 20
"""

import argparse
import statistics
import time

import torch
import torchvision.transforms as T
from PIL import Image

from caption_generator.data.dataset import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD
from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.caption_model import CaptionModel
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCLIP, EncoderCNN


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def load_model(checkpoint_path: str, device: str) -> CaptionModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = Vocabulary()
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = checkpoint["vocab_idx2word"]

    encoder_type = checkpoint.get("encoder_type", "resnet50")
    if encoder_type == "clip-vit-base-patch32":
        encoder = EncoderCLIP(fine_tune=False)
        decoder = DecoderWithAttention(vocab_size=len(vocab), encoder_dim=EncoderCLIP.OUTPUT_DIM)
    else:
        encoder = EncoderCNN(fine_tune=False)
        decoder = DecoderWithAttention(vocab_size=len(vocab))
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])

    return CaptionModel(encoder, decoder, vocab, device=device), encoder_type


def make_input(image_path: str, encoder_type: str) -> torch.Tensor:
    mean, std = (
        (CLIP_MEAN, CLIP_STD) if encoder_type == "clip-vit-base-patch32" else (IMAGENET_MEAN, IMAGENET_STD)
    )
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])
    return transform(Image.open(image_path).convert("RGB")).unsqueeze(0)


def time_calls(fn, n: int, warmup: int = 3) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)  # ms
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(0.95 * len(times)) - 1],
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--image", required=True, help="a single representative image, reused for every timed call"
    )
    parser.add_argument("--n", type=int, default=20, help="timed calls per device/mode")
    parser.add_argument("--devices", nargs="*", default=None, help="override auto-detected device list")
    args = parser.parse_args()

    devices = args.devices or available_devices()
    print(f"Devices to benchmark: {devices}\n")

    results = []
    for device in devices:
        model, encoder_type = load_model(args.checkpoint, device)
        image = make_input(args.image, encoder_type).to(device)

        greedy_stats = time_calls(lambda m=model, img=image: m.generate_greedy(img), args.n)
        beam_stats = time_calls(lambda m=model, img=image: m.generate_beam(img, beam_width=3), args.n)

        results.append((device, "greedy", greedy_stats))
        results.append((device, "beam(k=3)", beam_stats))

    print(
        f"{'device':<8} {'mode':<12} {'mean':>8} {'median':>8} {'p95':>8} {'min':>8} {'max':>8}  (ms/image)"
    )
    for device, mode, stats in results:
        print(
            f"{device:<8} {mode:<12} {stats['mean_ms']:>8.1f} {stats['median_ms']:>8.1f} "
            f"{stats['p95_ms']:>8.1f} {stats['min_ms']:>8.1f} {stats['max_ms']:>8.1f}"
        )


if __name__ == "__main__":
    main()
