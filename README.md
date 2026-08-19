# Multimodal Neural Image Caption Generator

[![CI](https://github.com/SovanDaraP02/image-caption-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/SovanDaraP02/image-caption-generator/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A CLIP ViT-B/32 encoder (ResNet-50/101/152 also supported, see Design decisions) plus Bahdanau attention plus an LSTM decoder, following *Show, Attend and Tell* (Xu et al., 2015). Trained in stages on progressively bigger data (Flickr8k, then COCO 50k, then COCO 113k) and a progressively better encoder (ResNet, then CLIP, then a fine-tuned CLIP). See Results for the full progression.

Given a photo, the model writes a caption for it and learns which part of the image to look at for each word it generates.

New here? [`WRITEUP.md`](./WRITEUP.md) is a short version of the Results section below. [`MODEL_CARD.md`](./MODEL_CARD.md) covers intended use, training data, and limitations in the standard model-card format.

## Live demo
https://image-caption-generator-bvudeamams3gdsghreqqjg.streamlit.app/

The app (`app.py`) has five captioning backends when you run it locally, but the public link only shows three (set by `PUBLIC_DEMO=true`, see `app.py`). The free hosting tier is ~1GB RAM, and two of the backends need more than that to even load. See "Why only three backends are public" below.

The model this project is actually about is the custom one: CLIP ViT-B/32 plus Bahdanau attention plus an LSTM decoder, trained from scratch on Flickr8k and COCO. It's the default option, and it's the one with real BLEU/CIDEr numbers in Results. Everything else in the picker is someone else's pretrained model, included for comparison, and labeled as such in the UI.

**Public on the live link:**
- **🎓 My trained model (default)** — this project's own architecture. Shorter and more generic than the options below, with occasional hallucination on unfamiliar scenes. That's the expected cost of training on ~113k images instead of hundreds of millions (see Limitations).
- **BLIP** — Salesforce's `blip-image-captioning-large` (~470M params). Short, accurate, fast even on CPU, free.
- **Claude** — calls the Anthropic API with a prompt asking for a full paragraph: objects, color and material, position, setting. The most detailed option, and the only one that reads like a hand-written photo description. Caption models like BLIP and this project's own model are trained on short reference captions and can't produce that style no matter how long you train them; that's a different task, closer to what instruction-tuned vision-language models do. Needs an `ANTHROPIC_API_KEY`, entered per visitor in the UI or set as an environment variable. Costs a small amount per image, paid by whoever enters the key, not by me — no key is preconfigured on the public deploy.

**Local only** (run `streamlit run app.py` yourself to try these):
- **BLIP-2** — Salesforce's `blip2-opt-2.7b` (~2.7B params). Richer captions than plain BLIP. Loading it on CPU peaks at around 14.6GB of RAM before quantization brings it down to about 3.3GB resident. That peak is what crashed the public deploy — not a guess, it happened. Full breakdown in `MODEL_CARD.md` and the `PUBLIC_DEMO` comment in `app.py`.
- **BLIP-3/xGen-MM** — Salesforce's ~4.6B-param, Phi-3-backed model. Instruction-tuned, so it can attempt the same detailed prompt as Claude, with weaker results. About an 18GB download. No free hosting tier has room for it.

### Why only three backends are public

Free hosting has real memory limits, and I'd rather run a smaller demo that works than a fuller one that crashes on people trying it. `DEPLOY_SPACES.md` covers a path to a BLIP-2-capable public deploy on Hugging Face Spaces, which has more RAM, but it needs a paid or verified compute tier that this deploy doesn't use.

All public backends accept multiple images at once; each gets its own caption.

## Architecture

```
Input image (3×224×224)
    → CLIP ViT-B/32 encoder (frozen, last block fine-tuned —
      or swap in ResNet-50/101/152, see Design decisions)
    → Feature map (49 regions × 768-d for CLIP, 2048-d for ResNet)
    → Attention ↔ LSTM decoder (one word at a time, attention
      recomputed at every step)
    → Generated caption
```

```mermaid
flowchart LR
    IMG["Image\n3×224×224"] --> ENC["EncoderCLIP\n(CLIP ViT-B/32, fine-tuned last block)"]
    ENC --> FEAT["Feature map\n49 regions × 768-d"]
    FEAT --> ATT["Attention\nBahdanau, per timestep"]
    HPREV["h(t-1)"] --> ATT
    ATT --> CTX["Context vector z(t)"]
    CTX --> LSTM["LSTMCell"]
    WORD["word embedding\n(t-1)"] --> LSTM
    LSTM --> HNEXT["h(t)"]
    HNEXT --> HPREV
    HNEXT --> FC["FC → vocab logits"]
    FC --> OUT["word(t)"]
```

## Design decisions

- **Bahdanau attention instead of Luong**: matches the reference paper, and it's more numerically stable at the hidden sizes used here (256-512).
- **LSTM decoder instead of a Transformer decoder**: for a project trained on a single GPU at Flickr8k scale, a recurrent decoder is cheaper to train and easier to debug end to end. A Transformer decoder with cross-attention to the 49 image tokens would be the natural next step with more data to justify it (see Roadmap).
- **Frozen encoder**: `EncoderCNN` and `EncoderCLIP` both stay frozen during most of training rather than fine-tuning end to end. On a dataset this size, unfreezing the whole network risks wrecking the pretrained features. Setting `fine_tune=True` on either one only unfreezes its last block (a residual block for ResNet, a transformer block for CLIP) as a middle ground.
- **CLIP encoder instead of ResNet** (see Results): I swapped this in after the 50k→113k ResNet run showed diminishing returns from more data alone, which raised a question — was the bottleneck data volume, or the encoder itself? ResNet's ImageNet-classification features aren't language-aligned. Re-running the same 113k images through CLIP confirmed it: +12.4% BLEU-4 on identical data. `EncoderCNN` is still in the codebase so this comparison stays reproducible.
- **Doubly stochastic attention regularization** (Xu et al. 2015, Section 4.2.1): the loss includes a term that pushes the model to spread attention across every image region over the course of a caption, not fixate on one spot. It measurably produces cleaner attention maps. See `attention_regularization` in `train.py`.
- **Teacher forcing during training**: the decoder sees the ground-truth previous word, not its own prediction. Trains faster and more stably, at the cost of a train/inference mismatch (exposure bias) that's a known limitation of this whole family of models, not something specific to this implementation.

## Results

`best_checkpoint.pth`, the checkpoint shipped in this repo and loaded by default in the "My trained model" backend, uses a CLIP ViT-B/32 encoder swapped in from the original ImageNet-pretrained ResNet and then fine-tuned (last transformer block unfrozen, low learning rate). Trained on the full 113,000-image COCO Karpathy train split, evaluated on a held-out 5,000-image test split it never saw:

| Metric | Score |
|---|---|
| BLEU-1 | 0.6879 |
| BLEU-2 | 0.5136 |
| BLEU-3 | 0.3671 |
| BLEU-4 | 0.2601 |
| CIDEr | 0.8083 |

Trained in two phases, both on a local Apple M4 Pro (MPS): 10 epochs with the encoder frozen (best at epoch 9, `val_loss=2.2332`), then 6 more epochs of fine-tuning warm-started from that checkpoint (best at epoch 5, `val_loss=2.1940`), via `scripts/train_local_coco.py --encoder clip-vit-base-patch32 --finetune-from <checkpoint>` — see "Three ways to train" below. METEOR was skipped; see Engineering notes for why.

I didn't save per-epoch logs from that original 113k run at the time — they just printed to a terminal. So here's a real curve from a smaller run instead: the same training loop, the same CLIP-frozen setup, 10,000 COCO images, 8 epochs, on the same machine.

![training loss curve: train and validation loss both decreasing smoothly over 8 epochs, from ~4.36/3.24 to ~2.77/2.51](assets/training_curve_demo.png)

Both curves drop smoothly with no divergence, which is what you want to see. The absolute numbers aren't directly comparable to the full run's `val_loss=2.1940` above since it's a shorter run on less data — it's here to show the shape of convergence, not to stand in for the real numbers.

Five checkpoints total, kept around for comparison (`best_checkpoint_flickr8k.pth`, `best_checkpoint_coco_50k.pth`, `best_checkpoint_coco_113k_resnet.pth`, `best_checkpoint_coco_clip_frozen.pth`, and the current fine-tuned one):

| Metric | Flickr8k (6k, ResNet) | COCO 50k (ResNet) | COCO 113k (ResNet) | COCO 113k (CLIP, frozen) | **COCO 113k (CLIP, fine-tuned)** |
|---|---|---|---|---|---|
| BLEU-1 | 0.5517 | 0.6363 | 0.6440 | 0.6789 | **0.6879** |
| BLEU-2 | 0.3737 | 0.4600 | 0.4650 | 0.5036 | **0.5136** |
| BLEU-3 | 0.2437 | 0.3188 | 0.3221 | 0.3564 | **0.3671** |
| BLEU-4 | 0.1585 | 0.2195 | 0.2215 | 0.2490 | **0.2601** |
| METEOR | 0.1953 | _skipped_ | _skipped_ | _skipped_ | _skipped_ |
| CIDEr | 0.4521 | 0.6680 | 0.6781 | 0.7729 | **0.8083** |

What these five runs actually show:

- **6k → 50k images, same ResNet encoder: a big improvement.** +38% BLEU-4, +48% CIDEr. Same code, more data, meaningfully better model — quality was clearly data-limited at this point.
- **50k → 113k images, same ResNet encoder: a much smaller improvement.** +0.9% BLEU-4, +1.5% CIDEr, with validation loss visibly flattening over the last three epochs. More than doubling the data bought a small, real gain — classic diminishing returns, and a sign that data volume wasn't the bottleneck anymore.
- **Same 113k images, ResNet → CLIP encoder: another big improvement.** +12.4% BLEU-4, +14.0% CIDEr — bigger than the entire 50k→113k jump. This is the direct test of the question the plateau raised: if more data wasn't helping, maybe the encoder was the limit instead. ResNet's ImageNet pretraining is 1,000-way object classification with no linguistic structure to it. CLIP is pretrained to match images directly to their captions (see `EncoderCLIP`'s docstring in `src/caption_generator/models/encoder.py`), so its features are already language-aligned before the decoder ever sees them. Every epoch scored better with CLIP than the equivalent ResNet epoch, and validation loss plateaued later.
- **Fine-tuning that CLIP encoder's last block: a smaller improvement.** +4.5% BLEU-4, +4.6% CIDEr over the frozen-CLIP run. Warm-started from the converged checkpoint, with the encoder's last block unfrozen at `1e-5` and the decoder continuing at `1e-4` — both well below the `4e-4` used for training from scratch, so a big early gradient couldn't wreck the pretrained CLIP features. Val loss improved through epoch 5, then ticked up slightly at epoch 6, suggesting this phase was near its own point of diminishing returns.

Put together, this is a plateau-hypothesis-test-result loop repeated four times: notice things stall, guess why, run the controlled comparison, see what actually happens. The 6k→113k→CLIP→fine-tuned progression is still far short of a production-scale model (see the BLIP/BLIP-2/BLIP-3/Claude backends in `app.py`) — 113k images is nothing next to the hundreds of millions those are trained on — but every step here was a measured improvement, not just more training for its own sake.

## Three ways to train

Same architecture, same `train_one_epoch`/`validate` loop, three places to run it depending on what hardware you have:

| | Flickr8k (baseline) | COCO via Colab/Kaggle | COCO via local script |
|---|---|---|---|
| Entry point | `notebooks/train_colab.ipynb` | `notebooks/train_colab_coco.ipynb` / `notebooks/train_kaggle_coco.ipynb` | `scripts/train_local_coco.py` |
| Training images | ~6,000 | up to 50,000 (configurable) | up to 113,000 (full Karpathy train split, configurable) |
| Split | ad-hoc 80/10/10 | Karpathy split | Karpathy split |
| Where it runs | Colab free-tier GPU | Colab/Kaggle free-tier GPU | your own machine (CUDA, Apple Silicon MPS, or CPU) |
| Runtime | ~30-60 min | several hours, subject to session limits | ~35-40 min/epoch at 50k images, ~80 min/epoch at 113k images, on an M4 Pro (MPS); no session limits, but ties up your machine for hours |
| Resumable | no | yes (Drive/Kaggle-output-backed) | yes (`data/coco/latest_checkpoint_coco.pth`) |

`scripts/train_local_coco.py` is for machines with real local acceleration (CUDA or Apple Silicon MPS), where free-tier session limits cost more time than they save. It downloads the same `yerevann/coco-karpathy` Hugging Face dataset the Colab/Kaggle notebooks use, caches the (image, caption) pairs to `data/coco/pairs_cache.json` so re-runs skip the download, and saves a checkpoint after every epoch — safe to interrupt with Ctrl-C, sleep, a crash, or an actual reboot, and pick back up.

```bash
python scripts/train_local_coco.py --encoder clip-vit-base-patch32 --n-train 113000 --n-val 5000 --n-test 5000  # train from scratch, frozen encoder
python scripts/train_local_coco.py --n-train 5000                                                                 # smaller/faster run, default ResNet encoder

# fine-tune an already-converged frozen-encoder checkpoint (what produced best_checkpoint.pth):
python scripts/train_local_coco.py --encoder clip-vit-base-patch32 --n-train 113000 --n-val 5000 --n-test 5000 \
    --finetune-from best_checkpoint_coco_clip.pth --epochs 6 --lr 1e-4 --encoder-lr 1e-5
```

`--encoder` picks between `resnet50` (default, `EncoderCNN`, ImageNet pretraining) and `clip-vit-base-patch32` (`EncoderCLIP`, CLIP's contrastive pretraining — see Results for why that mattered). Checkpoint filenames are suffixed by encoder (`best_checkpoint_coco_clip.pth` vs. `best_checkpoint_coco.pth`, plus `_finetuned` when `--finetune-from` is used), so runs don't clobber each other, and each checkpoint records which encoder produced it so `app.py`/`evaluate.py` load the right architecture and input normalization automatically.

`--finetune-from <checkpoint>` warm-starts both encoder and decoder from an existing checkpoint instead of starting from scratch, and unfreezes the encoder's last block with its own, much lower learning rate (`--encoder-lr`, default `1e-5`) than the decoder's (`--lr`, default `4e-4` — use something lower here too when fine-tuning, since the decoder's already converged and a from-scratch learning rate will overshoot). This is different from the automatic resume described below: resume continues the same run with the same hyperparameters, while `--finetune-from` starts a deliberately new phase on top of a finished run.

A few things I learned running this for real, over multiple unattended hours:

- **Keep the machine from sleeping mid-run.** Sleep pauses the process, but the wall-clock timer in the epoch log keeps counting through it, which looks like a huge slowdown but isn't. `caffeinate -i -w <pid>` on macOS handles this.
- **A reboot kills the run**, since it's a plain background process, not a daemon — but nothing is actually lost. The image cache and pairs cache survive on disk, so re-running just skips the download and resumes from the last saved epoch.
- **A checkpoint from a differently-sized run can't be resumed into a new run.** Vocab size depends on the training corpus, so `latest_checkpoint_coco.pth` from a 50k run has the wrong embedding/output shapes for a 113k run. The script now detects the vocab mismatch and starts fresh with a warning instead of crashing or silently loading corrupted weights — but it's cleaner to move or delete a previous run's `data/coco/latest_checkpoint_coco.pth` before starting a differently-sized one.
- If training looks much slower than expected with no errors, check for unrelated load on the machine before assuming the script is broken. Two things actually hit me here: macOS Spotlight indexing the freshly downloaded 113k-image dataset (`mediaanalysisd`/`spotlightknowledged` pinned at 80-97% CPU), fixed with a `data/coco/.metadata_never_index` marker file (a standard macOS mechanism, no `sudo` needed); and separately, memory pressure from having too many other apps open, which showed up as `vm.swapusage` near its ceiling and the training process burning way less CPU time than wall-clock time elapsed — it was mostly blocked on swapped-out memory, not actually computing. Closing browsers/IDEs/VM backends fixed it. Both are just the reality of running a multi-hour job on a machine you're also using for other things.
- **Use a much lower learning rate for the newly-unfrozen block than you'd use training from scratch.** The decoder here is already converged, and the encoder's last block has never seen a gradient for this task at all — a shared, from-scratch-sized learning rate risks a big early update wrecking the pretrained features instead of refining them. `--encoder-lr 1e-5` against a `--lr 1e-4` decoder (both below the `4e-4` used from scratch) worked here.

### Where the COCO run actually happens

`notebooks/train_colab_coco.ipynb` targets Colab, but Colab's free tier has undocumented, shifting session limits — Google's own FAQ says usage limits "vary over time" and doesn't publish them. In practice, sessions got reclaimed before a single epoch finished at this dataset size. Two things made that workable instead of a dealbreaker:

- The training loop saves full state (model, optimizer, scheduler, epoch number) after every epoch, backed up to Drive, and auto-resumes in a fresh session instead of restarting.
- The downloaded image set gets archived to Drive after a successful download, so reconnecting doesn't re-pay the ~35-minute download every time.

`notebooks/train_kaggle_coco.ipynb` runs the same training logic on Kaggle Notebooks instead, which publishes an actual quota (30 GPU-hours/week, up to 12 hours/session) and supports background execution (Save Version → Save & Run All) that survives closing the browser tab. It sidesteps the Colab problems by construction rather than working around them. Either notebook produces a checkpoint you can evaluate the same way — which one to use is about platform availability, not architecture.

## Attention visualizations

![attention heatmap: model correctly attends to the bus for "bus" and the building for "city"](assets/attention_visualizations/example_1.png)

_Generated with `visualize_attention.py` against `best_checkpoint.pth` (COCO 113k, fine-tuned CLIP encoder) on a real, held-out COCO validation photo. Caption: "a bus is driving down the street in a city." Each panel shows which region the model attended to while generating that word — "bus" lights up the bus itself, and "the"/"in"/"city" attend to the building, a reasonable stand-in for "urban setting" even though there's no single object to point at for that word._

## Caption examples

Two real outputs from `best_checkpoint.pth` on held-out COCO images. One clean, one wrong — not cherry-picked to only show the good side.

| | Caption |
|---|---|
| ![bathroom photo](assets/examples/good_bathroom.jpg) | **"a bathroom with a toilet and a shower"** — accurate. |
| ![cows in a dark barn](assets/examples/weak_cows.jpg) | **"two cows standing in a field with a yellow background"** — wrong. There's no field and nothing yellow in the photo; it's two cows in a dimly lit barn at night. Flickr8k and COCO both lean heavily toward cows photographed in daylight pasture scenes, so the model reached for that instead of describing the actual, visually unusual frame. |

## Limitations

- **Trained on ~113,000 images** — three or four orders of magnitude below what production vision-language models like BLIP train on (~14M pairs). Shorter, more generic captions are the expected result of that gap.
- Occasional repetitive captions on unfamiliar images, a known symptom of exposure bias in teacher-forced sequence models. Mitigated, not eliminated, by n-gram repetition blocking in `CaptionModel.generate_greedy`/`generate_beam` (see Engineering notes).
- **Hallucination on unfamiliar scenes.** See the cows example above. An earlier version of this model (Flickr8k-only, ResNet encoder) once captioned an empty storefront with no people as "a person is sitting on a sidewalk in front of a store." Same failure mode both times: faced with something outside its training distribution, the model falls back to its strongest learned prior instead of reporting what's actually there. That's a dataset-scale problem, not something decoding tricks can fix.
- Evaluated with greedy and beam-search decoding only — no length normalization, no diverse beam search.
- See [`MODEL_CARD.md`](./MODEL_CARD.md) for intended use, training data, and inference cost.

## Roadmap

Things I haven't done yet:

- **Fine-tune the CLIP encoder further, or unfreeze more of it.** The single-block fine-tune already done (see Results) helped, but showed early signs of its own diminishing returns by epoch 6. Unfreezing more blocks, or more epochs at a lower LR, is the natural next step if there's more to get out of it.
- **A Transformer decoder variant**, benchmarked against the current LSTM decoder on the same data and metrics.
- **Visual question answering is out of scope for this architecture.** Captioning and VQA are different tasks — VQA needs a text-question input and usually a different decoder — and doing it well would mean building on a pretrained vision-language model (BLIP-2, LLaVA) rather than extending this one.
- **Batch captioning in the Streamlit UI.** The model already supports batched inference; the demo just processes uploads one at a time in a loop instead of as a real batch.

## Serving the model

Two ways to run this outside the Streamlit demo:

**Local, with uvicorn:**
```bash
uvicorn api:app --reload --port 8000
curl -F "image=@assets/examples/good_bus.jpg" "http://localhost:8000/caption"
# optional: ?beam_width=3 for beam search instead of greedy decoding
```

**Containerized:**
```bash
docker build -t caption-generator-api .
docker run -p 8000:8000 caption-generator-api
```

`api.py` is a small FastAPI service (`POST /caption`, `GET /health`) around the same `CaptionModel` the Streamlit app and `evaluate.py` use. The checkpoint loads once at startup, not per request (see `api.py`'s `lifespan` handler), and downloads automatically the same way `app.py` does if it's missing locally. Tested in `tests/test_api.py` against a small untrained model, same approach as `test_caption_model.py`, so the tests stay fast and don't need the real checkpoint. See [`MODEL_CARD.md`](./MODEL_CARD.md) for measured latency (~20-45ms/image depending on device and decoding strategy).

## Engineering notes

- **Installable package.** `src/caption_generator/` is a real Python package (`pip install -e .`), not scripts glued together with `sys.path.insert`. Imports look like `from caption_generator.models.encoder import EncoderCNN` throughout — tests, scripts, and the app all use the same imports.
- **Configurable backbone.** `EncoderCNN(backbone="resnet101")` swaps the encoder with no downstream changes; every supported backbone outputs the same 2048-d feature vector per region (see `src/caption_generator/models/encoder.py`).
- **LR scheduling and early stopping.** `train.py` halves the learning rate when validation loss plateaus for 2 epochs, and stops after 4 epochs with no improvement, instead of running a fixed number of epochs.
- **N-gram repetition blocking at decode time.** `generate_greedy` and `generate_beam` skip any candidate word that would recreate a 3-gram already generated (`no_repeat_ngram_size=3`, matching HuggingFace's `generate()` convention). This removes visible loops like "a white dress and a white dress" without retraining. It doesn't and can't fix factual accuracy, just decoding-level repetition (see Limitations).
- **METEOR is optional in `evaluate()`** (`skip_meteor=True`). `pycocoevalcap`'s METEOR scorer shells out to a Java subprocess and parses its output line by line; in practice it sometimes throws parsing errors, and after a failure the subprocess can hang during cleanup instead of failing fast. BLEU and CIDEr don't have this dependency and aren't affected. It's a tooling issue, not a modeling one.
- **Actually tested.** `tests/` covers every module — shapes, gradient flow, vocabulary edge cases, dataset batching and augmentation, greedy/beam decoding, n-gram blocking — plus the FastAPI service (`test_api.py`). Runs in a few seconds, no GPU or dataset download needed. CI runs it on every push.
- **Linting, formatting, type-checking.** `ruff check`/`ruff format` and `mypy` are set up in `pyproject.toml` (`pip install -e ".[dev]"` installs all of it). Both are actually clean, including across the `EncoderCNN`/`EncoderCLIP` union types that CLIP support introduced.
- **Inference latency, measured.** `scripts/benchmark_inference.py` times greedy and beam-search decoding on every available device (CPU, MPS, CUDA). See [`MODEL_CARD.md`](./MODEL_CARD.md) for the numbers from the machine training ran on.

## Setup

```bash
git clone <this-repo>
cd image-caption-generator
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"   # installs the package + pytest/ruff/mypy, editable
pytest                     # verify everything works, a few seconds
ruff check . && mypy src/caption_generator   # lint + type-check, both clean
```

See [`SETUP_DATA.md`](./SETUP_DATA.md) for downloading Flickr8k, and [`notebooks/train_colab.ipynb`](./notebooks/train_colab.ipynb) for a ready-to-run Colab notebook that does the whole pipeline on a free GPU. For a larger COCO subset, use [`notebooks/train_colab_coco.ipynb`](./notebooks/train_colab_coco.ipynb) (Colab) or [`notebooks/train_kaggle_coco.ipynb`](./notebooks/train_kaggle_coco.ipynb) (Kaggle — worth it if Colab's session limits get in the way).

```bash
python -m caption_generator.train                 # trains, saves best_checkpoint.pth
python -m caption_generator.evaluate               # BLEU / METEOR / CIDEr on the test split
python -m caption_generator.visualize_attention \
    --checkpoint best_checkpoint.pth --image path/to/image.jpg
python scripts/benchmark_inference.py \
    --checkpoint best_checkpoint.pth --image path/to/image.jpg   # measure real latency on this machine
streamlit run app.py                                # local web demo
uvicorn api:app --reload --port 8000                # JSON API instead — see "Serving the model"
```

See [`DEPLOY_STREAMLIT.md`](./DEPLOY_STREAMLIT.md) to put the demo on a public URL for free, or [`DEPLOY_SPACES.md`](./DEPLOY_SPACES.md) for a Hugging Face Spaces alternative.

## Project structure

```
image-caption-generator/
├── pyproject.toml             # package metadata + dependencies (source of truth)
├── src/caption_generator/
│   ├── data/
│   │   ├── vocabulary.py      # Vocabulary class, tokenization
│   │   └── dataset.py          # ImageCaptionDataset + collate_fn
│   ├── models/
│   │   ├── encoder.py           # EncoderCNN (ResNet-50/101/152) + EncoderCLIP (CLIP ViT-B/32)
│   │   ├── attention.py         # Bahdanau attention
│   │   ├── decoder.py            # LSTM decoder with attention
│   │   └── caption_model.py     # composed model + greedy/beam inference
│   ├── train.py
│   ├── evaluate.py
│   └── visualize_attention.py
├── tests/                     # pytest suite, no GPU/dataset needed (incl. test_api.py)
├── notebooks/
│   ├── train_colab.ipynb        # Flickr8k baseline training notebook (Colab)
│   ├── train_colab_coco.ipynb   # larger, more diverse COCO training notebook (Colab)
│   └── train_kaggle_coco.ipynb  # same COCO training, adapted for Kaggle's longer/more predictable sessions
├── scripts/
│   ├── train_local_coco.py    # COCO training on local CUDA/MPS hardware, --encoder resnet50|clip-vit-base-patch32
│   └── benchmark_inference.py # measures real greedy/beam-search latency per device
├── assets/
│   ├── attention_visualizations/  # real heatmap output from visualize_attention.py
│   └── examples/                   # real good/weak caption examples, referenced in README + MODEL_CARD
├── .github/workflows/ci.yml  # runs pytest + ruff + mypy on every push
├── app.py                    # Streamlit demo (imports the installed package)
├── api.py                    # FastAPI JSON service (POST /caption) — see "Serving the model"
├── Dockerfile                 # containerizes api.py
├── MODEL_CARD.md
├── WRITEUP.md                 # short, non-technical version of Results
├── SETUP_DATA.md
├── DEPLOY_SPACES.md
└── requirements.txt           # mirrors pyproject.toml, needed by Hugging Face Spaces
```

## Acknowledgments

Architecture based on Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* (2015). Implementation informed by [sgrvinod's PyTorch tutorial on image captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning), adapted to current PyTorch/torchvision APIs and rewritten independently.
