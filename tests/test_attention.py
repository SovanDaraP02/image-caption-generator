import torch

from caption_generator.models.attention import Attention


def test_shapes():
    B, num_regions, encoder_dim, decoder_dim = 2, 49, 2048, 512
    attention = Attention(encoder_dim=encoder_dim, decoder_dim=decoder_dim)

    context, alpha = attention(torch.randn(B, num_regions, encoder_dim),
                                torch.randn(B, decoder_dim))

    assert context.shape == (B, encoder_dim)
    assert alpha.shape == (B, num_regions)


def test_weights_sum_to_one():
    B, num_regions, encoder_dim, decoder_dim = 4, 49, 2048, 512
    attention = Attention(encoder_dim=encoder_dim, decoder_dim=decoder_dim)

    _, alpha = attention(torch.randn(B, num_regions, encoder_dim),
                          torch.randn(B, decoder_dim))

    assert torch.allclose(alpha.sum(dim=1), torch.ones(B), atol=1e-5)


def test_weights_are_nonnegative():
    attention = Attention(encoder_dim=2048, decoder_dim=512)
    _, alpha = attention(torch.randn(3, 49, 2048), torch.randn(3, 512))
    assert (alpha >= 0).all()


def test_context_is_weighted_average_of_regions():
    """When one region gets all the attention weight, the context vector
    should equal that region's feature vector exactly."""
    encoder_dim = 8
    encoder_out = torch.randn(1, 49, encoder_dim)

    attention = Attention(encoder_dim=encoder_dim, decoder_dim=4)
    # bypass the learned scoring and force a one-hot distribution directly
    fake_alpha = torch.zeros(1, 49)
    fake_alpha[0, 5] = 1.0
    context = (encoder_out * fake_alpha.unsqueeze(2)).sum(dim=1)

    assert torch.allclose(context.squeeze(0), encoder_out[0, 5])
