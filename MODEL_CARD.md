# Model Card: Image Caption Generator

Following the structure of Mitchell et al., *Model Cards for Model Reporting* (2019). Covers `best_checkpoint.pth`, the checkpoint loaded by default in `app.py` and `api.py`.

## Model details

- **Architecture**: CLIP ViT-B/32 vision encoder (`EncoderCLIP`), Bahdanau additive attention, single-layer LSTM decoder, following Xu et al., *Show, Attend and Tell* (2015). See `src/caption_generator/models/`.
- **Trained from scratch**: the attention module and decoder (embedding, LSTM cell, output projection). The encoder is pretrained (OpenAI CLIP), with only its last transformer block fine-tuned.
- **Vocabulary**: 9,956 tokens, built from the COCO Karpathy training-split captions (`min_word_freq=5`).
- **Input**: RGB image, resized to 224×224, normalized with CLIP's mean/std.
- **Output**: one caption, generated word by word (greedy or beam search, `beam_width` configurable).
- **Developed by**: an individual project, not a company or team release.
- **License**: MIT (see `LICENSE`), which covers the code. The model was trained on COCO, which has its own terms — non-commercial research use, see [cocodataset.org/#termsofuse](https://cocodataset.org/#termsofuse).

## Intended use

- **In scope**: a portfolio and research demonstration of attention-based image captioning, built and evaluated end to end by one person; a reference implementation for how encoder-attention-decoder captioning works; a base for further experimentation (more data, a different decoder — see the README's Roadmap).
- **Out of scope**: production photo captioning at scale, or any safety-critical or accessibility-critical alt-text use. See Limitations for why — hallucination risk makes this unsuitable without a human checking the output. Don't treat the captions as guaranteed accurate.

## Training data

- **Primary**: MS COCO, Karpathy split. Up to 113,000 training images, 5,000 validation, 5,000 test, each with 5 human-written reference captions.
- **Earlier runs, kept for comparison**: Flickr8k (~6,000 training images), a 50k-image COCO subset. Neither is part of the final model — they're separate checkpoints used in the README's ablation comparison.
- No filtering was done for demographic balance, content moderation, or bias beyond whatever COCO itself already applies. The captions and imagery carry whatever biases exist in COCO's underlying Flickr photos and MTurk-written captions.

## Training procedure

- Two phases: 10 epochs with the CLIP encoder frozen (Adam, decoder LR `4e-4`), then 6 more epochs fine-tuning just the encoder's last transformer block (`encoder LR 1e-5`, `decoder LR 1e-4`), warm-started from the frozen-encoder checkpoint.
- Loss: token-level cross-entropy (padding excluded) plus a doubly-stochastic attention regularization term (Xu et al. 2015, §4.2.1).
- Learning rate halved when validation loss plateaus for 2 epochs; training stops after 4 epochs with no improvement. The checkpoint saved is whichever epoch had the best validation loss, not necessarily the last one.
- Hardware: a local Apple M4 Pro (MPS backend), no cloud GPU for this run. About 80 minutes per epoch at 113k images.
- Full details — the three interchangeable training environments (Colab, Kaggle, local) and what actually went wrong running this unattended for hours — are in the main [README](./README.md#three-ways-to-train).
- Per-epoch loss curve: the original 113k run's logs weren't saved, so the [README's Results section](./README.md#results) includes a curve from a smaller 10k-image, 8-epoch run instead, same training loop and architecture, labeled as such.

## Evaluation

Held-out COCO test split (5,000 images, never seen in training), scored with `pycocoevalcap`, the library used in most captioning papers:

| Metric | Score |
|---|---|
| BLEU-1 | 0.6879 |
| BLEU-2 | 0.5136 |
| BLEU-3 | 0.3671 |
| BLEU-4 | 0.2601 |
| CIDEr | 0.8083 |

METEOR wasn't run for this checkpoint — a tooling issue, not a modeling one, explained in the README's Engineering notes. The full comparison across all 5 trained checkpoints is in the README's [Results](./README.md#results) section.

## Inference cost

Measured with `scripts/benchmark_inference.py` on the same M4 Pro used for training (15 timed calls after 3 warmup calls, one 224×224 image, encoder and decoder forward pass both included):

| Device | Decoding | Mean | Median | p95 |
|---|---|---|---|---|
| CPU | greedy | 31.8ms | 31.2ms | 33.1ms |
| CPU | beam (k=3) | 46.3ms | 37.0ms | 76.9ms |
| MPS | greedy | 20.6ms | 19.9ms | 23.2ms |
| MPS | beam (k=3) | 45.9ms | 43.3ms | 53.6ms |

This excludes the one-time checkpoint load (~380MB from disk) and CLIP weight initialization, which happen once at process startup (see `api.py`'s `lifespan` handler), not per request.

## Known limitations

- **Object hallucination on unfamiliar scenes.** A night-time photo of cows in a dark barn was captioned "two cows standing in a field with a yellow background" — there's no field, and nothing yellow, in the photo. The model falls back to a strong training-data pattern (cows in daylight pasture) when the actual scene doesn't match anything it's seen much of. See `assets/examples/weak_cows.jpg`.
- **Exposure bias.** Training uses teacher forcing (the model always sees the correct previous word), so errors at inference time can compound in ways training loss doesn't capture. Mitigated but not solved by n-gram repetition blocking at decode time.
- **Trained on ~113k images**, three or four orders of magnitude below production vision-language models (BLIP trains on ~14M pairs). Shorter, more generic captions follow directly from that.
- **No content filtering or fairness auditing** was done on generated captions beyond what COCO's own upstream curation already applies.

## How to reproduce these numbers

```bash
python -m caption_generator.evaluate           # BLEU/CIDEr on the held-out test split
python scripts/benchmark_inference.py --checkpoint best_checkpoint.pth --image assets/examples/good_bus.jpg
```
