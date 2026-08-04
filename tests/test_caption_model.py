import torch

from vocabulary import Vocabulary
from encoder import EncoderCNN
from decoder import DecoderWithAttention
from caption_model import CaptionModel


def _build_model():
    vocab = Vocabulary(min_word_freq=1).build([
        "a dog runs in the grass", "a cat sits on the mat",
    ])
    encoder = EncoderCNN(fine_tune=False, pretrained=False)
    decoder = DecoderWithAttention(vocab_size=len(vocab), encoder_dim=2048)
    return CaptionModel(encoder, decoder, vocab, device="cpu", max_len=10), vocab


def test_greedy_generation_runs_end_to_end():
    model, _ = _build_model()
    dummy_image = torch.randn(1, 3, 224, 224)

    caption, alphas = model.generate_greedy(dummy_image)

    assert isinstance(caption, str)
    assert len(alphas) <= model.max_len
    assert all(a.shape == (49,) for a in alphas)


def test_beam_search_runs_end_to_end():
    model, _ = _build_model()
    dummy_image = torch.randn(1, 3, 224, 224)

    caption = model.generate_beam(dummy_image, beam_width=3)

    assert isinstance(caption, str)


def test_beam_search_never_emits_start_or_end_tokens():
    model, vocab = _build_model()
    dummy_image = torch.randn(1, 3, 224, 224)

    caption = model.generate_beam(dummy_image, beam_width=3)
    tokens = caption.split()

    assert vocab.START_TOKEN not in tokens
    assert vocab.END_TOKEN not in tokens
