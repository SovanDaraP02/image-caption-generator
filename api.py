"""FastAPI service exposing the trained captioning model as a JSON HTTP
API, for programmatic/integration use -- as opposed to app.py's
Streamlit UI, which is built for a human clicking around in a browser.
Same underlying CaptionModel and checkpoint; this just gives another
service a way to call it (POST an image, get a caption back) without
scraping a web page.

Run locally:
    uvicorn api:app --reload --port 8000

Try it:
    curl -F "image=@assets/examples/good_bus.jpg" \\
         "http://localhost:8000/caption?beam_width=3"

The checkpoint is loaded once at process startup (see `lifespan` below),
not per-request -- reloading a ~380MB checkpoint on every call would make
latency dominated by disk I/O instead of the ~20-45ms/image the model
itself costs (see scripts/benchmark_inference.py).
"""

import io
import os
from contextlib import asynccontextmanager
from typing import Annotated

import requests
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from caption_generator.data.dataset import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD
from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.caption_model import CaptionModel
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCLIP, EncoderCNN

CHECKPOINT_PATH = os.environ.get("CAPTION_CHECKPOINT_PATH", "best_checkpoint.pth")
CHECKPOINT_DOWNLOAD_URL = (
    "https://huggingface.co/sovandara6262/image-caption-generator-checkpoint/resolve/main/best_checkpoint.pth"
)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB: generous for a photo, cheap guard against abuse

_state: dict[str, object] = {}


def _download_checkpoint_if_missing(path: str) -> None:
    if os.path.exists(path):
        return
    response = requests.get(CHECKPOINT_DOWNLOAD_URL, stream=True, timeout=60)
    response.raise_for_status()
    tmp_path = path + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
    os.rename(tmp_path, path)  # atomic: no request sees a half-written file


def _load_model(checkpoint_path: str) -> tuple[CaptionModel, str]:
    _download_checkpoint_if_missing(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    vocab = Vocabulary()
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = checkpoint["vocab_idx2word"]

    encoder_type = checkpoint.get("encoder_type", "resnet50")
    encoder: EncoderCNN | EncoderCLIP
    if encoder_type == "clip-vit-base-patch32":
        encoder = EncoderCLIP(fine_tune=False)
        decoder = DecoderWithAttention(vocab_size=len(vocab), encoder_dim=EncoderCLIP.OUTPUT_DIM)
    else:
        encoder = EncoderCNN(fine_tune=False)
        decoder = DecoderWithAttention(vocab_size=len(vocab))
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])

    return CaptionModel(encoder, decoder, vocab, device="cpu"), encoder_type


@asynccontextmanager
async def lifespan(_: FastAPI):
    model, encoder_type = _load_model(CHECKPOINT_PATH)
    mean, std = (CLIP_MEAN, CLIP_STD) if encoder_type == "clip-vit-base-patch32" else (IMAGENET_MEAN, IMAGENET_STD)
    _state["model"] = model
    _state["transform"] = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])
    _state["encoder_type"] = encoder_type
    yield
    _state.clear()


app = FastAPI(
    title="Image Caption Generator API",
    description="CNN/CLIP encoder + Bahdanau attention + LSTM decoder, trained from scratch. "
    "See the project README for architecture, training, and evaluation details.",
    lifespan=lifespan,
)


class CaptionResponse(BaseModel):
    caption: str
    encoder: str
    decoding: str


class HealthResponse(BaseModel):
    status: str
    encoder: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", encoder=str(_state["encoder_type"]))


@app.post("/caption", response_model=CaptionResponse)
async def caption(
    image: Annotated[UploadFile, File(description="JPEG/PNG image to caption")],
    beam_width: int = 0,
) -> CaptionResponse:
    """beam_width=0 (default) uses greedy decoding, the same latency profile
    benchmarked in scripts/benchmark_inference.py. beam_width>=2 uses beam
    search instead -- slower, usually a better caption (see README, "Turning
    predictions into a sentence")."""
    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}") from e

    model: CaptionModel = _state["model"]  # type: ignore[assignment]
    transform: T.Compose = _state["transform"]  # type: ignore[assignment]
    tensor = transform(pil_image).unsqueeze(0)

    if beam_width and beam_width >= 2:
        text = model.generate_beam(tensor, beam_width=beam_width)
        decoding = f"beam(k={beam_width})"
    else:
        text, _ = model.generate_greedy(tensor)
        decoding = "greedy"

    return CaptionResponse(caption=text, encoder=str(_state["encoder_type"]), decoding=decoding)
