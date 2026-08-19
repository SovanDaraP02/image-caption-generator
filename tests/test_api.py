import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api
from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.caption_model import CaptionModel
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCNN


def _tiny_model() -> tuple[CaptionModel, str]:
    """Same untrained, synthetic-vocab construction test_caption_model.py
    uses -- fast, no checkpoint download or network access, and enough to
    exercise the API's request/response plumbing end-to-end."""
    vocab = Vocabulary(min_word_freq=1).build(["a dog runs in the grass"])
    encoder = EncoderCNN(fine_tune=False, pretrained=False)
    decoder = DecoderWithAttention(vocab_size=len(vocab), encoder_dim=2048)
    return CaptionModel(encoder, decoder, vocab, device="cpu", max_len=8), "resnet50"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_load_model", lambda checkpoint_path: _tiny_model())
    with TestClient(api.app) as c:
        yield c


def _fake_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 180, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "encoder": "resnet50"}


def test_caption_returns_greedy_by_default(client):
    response = client.post("/caption", files={"image": ("test.jpg", _fake_jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["caption"], str)
    assert body["decoding"] == "greedy"
    assert body["encoder"] == "resnet50"


def test_caption_uses_beam_search_when_requested(client):
    response = client.post(
        "/caption?beam_width=3", files={"image": ("test.jpg", _fake_jpeg_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["decoding"] == "beam(k=3)"


def test_caption_rejects_non_image_upload(client):
    response = client.post(
        "/caption", files={"image": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_caption_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/caption", files={"image": ("test.jpg", _fake_jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 413
