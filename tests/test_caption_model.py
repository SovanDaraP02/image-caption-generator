import torch

from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.caption_model import CaptionModel, _would_repeat_ngram
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCNN


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


def test_would_repeat_ngram_blocks_exact_repeat():
    # sequence [1, 2, 3, 1, 2] followed by 3 would recreate the trigram [1, 2, 3]
    assert _would_repeat_ngram([1, 2, 3, 1, 2], next_id=3, n=3) is True


def test_would_repeat_ngram_allows_novel_continuation():
    assert _would_repeat_ngram([1, 2, 3, 1, 2], next_id=9, n=3) is False


def test_would_repeat_ngram_disabled_when_n_is_zero():
    assert _would_repeat_ngram([1, 2, 3, 1, 2], next_id=3, n=0) is False


def test_would_repeat_ngram_no_false_positive_before_enough_history():
    # can't form a trigram at all yet with only 1 prior token
    assert _would_repeat_ngram([1], next_id=1, n=3) is False


def test_greedy_generation_never_emits_unk():
    # untrained (random-init) model, so <unk> is a plausible top logit at
    # some step by chance -- this exercises the blocking logic for real,
    # not just when it happens not to matter.
    model, vocab = _build_model()
    dummy_image = torch.randn(1, 3, 224, 224)

    caption, _ = model.generate_greedy(dummy_image)

    assert vocab.UNK_TOKEN not in caption.split()


def test_beam_search_never_emits_unk():
    model, vocab = _build_model()
    dummy_image = torch.randn(1, 3, 224, 224)

    caption = model.generate_beam(dummy_image, beam_width=3)

    assert vocab.UNK_TOKEN not in caption.split()
