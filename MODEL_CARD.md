# Model Card: Image Caption Generator

Following the structure of Mitchell et al., *Model Cards for Model Reporting* (2019). Covers `best_checkpoint.pth`, the checkpoint loaded by default in `app.py` and `api.py`.

## Model details

- **Architecture**: CLIP ViT-B/32 vision encoder (`EncoderCLIP`) + Bahdanau additive attention + single-layer LSTM decoder, following Xu et al., *Show, Attend and Tell* (2015). See `src/caption_generator/models/`.
- **Parameters trained from scratch**: attention module, decoder (embedding, LSTM cell, output projection) — the encoder is pretrained (OpenAI CLIP) with only its last transformer block fine-tuned.
- **Vocabulary**: 9,956 tokens, built from the COCO Karpathy training-split captions (`min_word_freq=5`).
- **Input**: RGB image, resized to 224×224, normalized with CLIP's mean/std.
- **Output**: a single free-text caption, generated word-by-word (greedy or beam search, `beam_width` configurable).
- **Developed by**: an individual project, not a company/team release.
- **License**: MIT (see `LICENSE`) — applies to the code; the model was trained on COCO, which has its own terms (non-commercial research use — see [cocodataset.org/#termsofuse](https://cocodataset.org/#termsofuse)).

## Intended use

- **In scope**: portfolio/research demonstration of an attention-based image captioning architecture, built and evaluated end-to-end by one person; a reference implementation for learning how encoder-attention-decoder captioning works; a base for further experimentation (larger data, decoder architecture changes — see README's Roadmap).
- **Out of scope**: production photo-captioning at scale, safety-critical or accessibility-critical alt-text generation (see Limitations — hallucination risk makes this unsuitable without human review), any use implying the captions are guaranteed factually accurate.

## Training data

- **Primary**: MS COCO, Karpathy split, up to 113,000 training images, 5,000 validation, 5,000 test — each image with 5 human-written reference captions.
- **Earlier-stage / comparison runs**: Flickr8k (~6,000 training images), COCO 50k-image subset — kept as separate checkpoints for the ablation comparison in the README, not used in the final model.
- Data was **not** filtered for demographic balance, content moderation, or bias beyond what COCO itself applies upstream — captions and imagery reflect whatever biases exist in COCO's underlying Flickr-sourced photos and MTurk-written captions.

## Training procedure

- Two phases: 10 epochs with the CLIP encoder frozen (Adam, decoder LR `4e-4`), then 6 further epochs fine-tuning only the encoder's last transformer block (`encoder LR 1e-5`, `decoder LR 1e-4`), warm-started from the frozen-encoder checkpoint.
- Loss: token-level cross-entropy (padding excluded) + a doubly-stochastic attention regularization term (Xu et al. 2015, §4.2.1).
- LR halved on a 2-epoch validation-loss plateau; training stopped after 4 epochs with no improvement. Best checkpoint by validation loss, not final epoch.
- Hardware: local Apple M4 Pro (MPS backend), no cloud GPU for this final run. ~80 min/epoch at 113k images.
- Full details, including the three interchangeable training environments (Colab / Kaggle / local) and real problems hit running this unattended for hours, are in the main [README](./README.md#three-ways-to-train).
- Per-epoch loss curve: the original 113k run's logs weren't captured, so the [README's Results section](./README.md#results) includes a real curve from a smaller 10k-image/8-epoch demonstration run instead, same training loop and architecture, clearly labeled as such.

## Evaluation

Held-out COCO test split (5,000 images, never seen in training), scored with `pycocoevalcap` (the standard library used in captioning papers):

| Metric | Score |
|---|---|
| BLEU-1 | 0.6879 |
| BLEU-2 | 0.5136 |
| BLEU-3 | 0.3671 |
| BLEU-4 | 0.2601 |
| CIDEr | 0.8083 |

METEOR was not run for this checkpoint (documented tooling issue, not a modeling one — see README "Engineering notes"). Full comparison across all 5 trained checkpoints (data-scale and encoder ablations) is in the README's [Results](./README.md#results) section.

## Inference cost

Measured with `scripts/benchmark_inference.py` on the same M4 Pro used for training (n=15 timed calls, after 3 warmup calls, single 224×224 image, encoder+decoder forward pass included):

| Device | Decoding | Mean | Median | p95 |
|---|---|---|---|---|
| CPU | greedy | 31.8ms | 31.2ms | 33.1ms |
| CPU | beam (k=3) | 46.3ms | 37.0ms | 76.9ms |
| MPS | greedy | 20.6ms | 19.9ms | 23.2ms |
| MPS | beam (k=3) | 45.9ms | 43.3ms | 53.6ms |

Excludes one-time checkpoint load (~380MB from disk) and CLIP weight initialization, which happen once at process startup (see `api.py`'s `lifespan` handler), not per request.

## Known limitations

- **Object hallucination on out-of-distribution scenes**: e.g. a night-time/dark-barn photo of cows was captioned "two cows standing in a field with a yellow background" — no field, no yellow anything is present. The model falls back to a strong training-data prior (cows ≈ pasture scenes) when the actual scene doesn't match anything well-represented in training data. See `assets/examples/weak_cows.jpg`.
- **Exposure bias**: trained with teacher forcing (always shown the correct previous word), so inference-time errors can compound in ways training loss doesn't capture. Partially mitigated, not eliminated, by n-gram repetition blocking at decode time.
- **Trained on ~113k images**, roughly 3-4 orders of magnitude less than production vision-language models (BLIP: ~14M pairs) — captions are shorter and more generic than those systems produce, by design/necessity, not a bug.
- **No content filtering or fairness auditing was performed** on generated captions beyond what's inherited from COCO's own upstream curation.

## How to reproduce these numbers

```bash
python -m caption_generator.evaluate           # BLEU/CIDEr on the held-out test split
python scripts/benchmark_inference.py --checkpoint best_checkpoint.pth --image assets/examples/good_bus.jpg
```
