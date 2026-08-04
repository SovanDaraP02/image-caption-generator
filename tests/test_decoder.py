import torch
import torch.nn as nn

from decoder import DecoderWithAttention


def test_forward_shapes():
    B, num_regions, encoder_dim, vocab_size, T = 2, 49, 2048, 50, 7
    decoder = DecoderWithAttention(vocab_size=vocab_size, encoder_dim=encoder_dim)

    encoder_out = torch.randn(B, num_regions, encoder_dim)
    captions = torch.randint(0, vocab_size, (B, T))

    logits, alphas = decoder(encoder_out, captions)

    assert logits.shape == (B, T - 1, vocab_size)
    assert alphas.shape == (B, T - 1, num_regions)


def test_loss_computes_without_error():
    B, num_regions, encoder_dim, vocab_size, T = 2, 49, 2048, 50, 7
    decoder = DecoderWithAttention(vocab_size=vocab_size, encoder_dim=encoder_dim)

    encoder_out = torch.randn(B, num_regions, encoder_dim)
    captions = torch.randint(0, vocab_size, (B, T))
    logits, _ = decoder(encoder_out, captions)

    targets = captions[:, 1:]
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), targets.reshape(-1))
    assert torch.isfinite(loss)


def test_init_hidden_state_shapes():
    encoder_dim, decoder_dim = 2048, 512
    decoder = DecoderWithAttention(vocab_size=50, encoder_dim=encoder_dim, decoder_dim=decoder_dim)
    encoder_out = torch.randn(3, 49, encoder_dim)

    h, c = decoder.init_hidden_state(encoder_out)
    assert h.shape == (3, decoder_dim)
    assert c.shape == (3, decoder_dim)


def test_gradients_flow_to_decoder_params():
    """A regression guard for the #1 beginner bug in this project (per the
    blueprint's debugging checklist): a shape mismatch that silently breaks
    backprop rather than raising, leaving trainable params with .grad=None."""
    decoder = DecoderWithAttention(vocab_size=20, encoder_dim=64, decoder_dim=32, attention_dim=16)
    encoder_out = torch.randn(2, 49, 64)
    captions = torch.randint(0, 20, (2, 5))

    logits, _ = decoder(encoder_out, captions)
    loss = logits.sum()
    loss.backward()

    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
