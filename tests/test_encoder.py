import pytest
import torch

from encoder import EncoderCNN


def test_output_shape():
    encoder = EncoderCNN(fine_tune=False, pretrained=False)
    dummy_images = torch.randn(2, 3, 224, 224)
    out = encoder(dummy_images)
    assert out.shape == (2, 49, 2048)


def test_frozen_by_default():
    encoder = EncoderCNN(fine_tune=False, pretrained=False)
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert n_trainable == 0


def test_fine_tune_unfreezes_last_block_only():
    encoder = EncoderCNN(fine_tune=True, pretrained=False)
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in encoder.parameters())
    assert 0 < n_trainable < n_total


@pytest.mark.parametrize("backbone", ["resnet50", "resnet101", "resnet152"])
def test_alternate_backbones_share_output_shape(backbone):
    """All supported backbones must be drop-in swaps -- same (B, 49, 2048)
    output -- since nothing downstream (attention/decoder dims) adapts to
    a different channel count."""
    encoder = EncoderCNN(fine_tune=False, pretrained=False, backbone=backbone)
    out = encoder(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 49, 2048)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        EncoderCNN(pretrained=False, backbone="not-a-real-backbone")
