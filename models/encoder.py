"""
EncoderCNN: wraps a pretrained ResNet-50, strips the classification head,
and returns spatial feature maps instead of a single pooled vector.

Day 5 concept check: WHY keep the spatial grid instead of one vector?
Because the attention mechanism needs to choose *where* to look for each
word. A single pooled vector has already thrown that spatial information
away — you can't un-pool it. Keeping the 7x7 grid (49 regions) is what
makes "attention" possible at all.
"""

import torch
import torch.nn as nn
import torchvision


class EncoderCNN(nn.Module):
    def __init__(self, fine_tune: bool = False):
        super().__init__()

        # Try pretrained ImageNet weights first (needs internet access to
        # download.pytorch.org -- works on Colab, may not work in an
        # offline sandbox). Falls back to random init so this module is
        # still importable/testable everywhere; training quality depends
        # on actually getting the pretrained weights, so check the
        # printed message below.
        try:
            resnet = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            )
        except Exception as e:
            print(f"[EncoderCNN] Could not download pretrained weights ({e}). "
                  f"Falling back to random init -- fine for shape self-tests, "
                  f"NOT fine for real training.")
            resnet = torchvision.models.resnet50(weights=None)

        # Strip the last two layers: adaptive avg pool + fc classifier.
        # Everything up to and including layer4 gives us the spatial
        # feature map we want.
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 224, 224)
        returns: (B, 49, 2048)  -- 49 spatial regions, each 2048-dim
        """
        features = self.resnet(images)              # (B, 2048, 7, 7)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)      # (B, 7, 7, 2048)
        features = features.reshape(B, H * W, C)      # (B, 49, 2048)
        return features

    def fine_tune(self, fine_tune: bool = False):
        """Freeze everything by default (Week/Day 5 default). Only if you
        later choose to fine-tune do we unfreeze the last conv block —
        never unfreeze the whole network on a dataset this small, you'll
        destroy the pretrained features (catastrophic forgetting)."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        if fine_tune:
            # unfreeze only layer4 (the last residual block)
            for layer in list(self.resnet.children())[7:]:
                for param in layer.parameters():
                    param.requires_grad = True


if __name__ == "__main__":
    # shape self-test — run with: python encoder.py
    encoder = EncoderCNN(fine_tune=False)
    dummy_images = torch.randn(2, 3, 224, 224)  # batch of 2 fake images
    out = encoder(dummy_images)
    print(f"Output shape: {tuple(out.shape)}  (expected: (2, 49, 2048))")
    assert out.shape == (2, 49, 2048), "encoder output shape is wrong"

    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Trainable params (should be 0, frozen by default): {n_trainable}")
    assert n_trainable == 0

    print("encoder.py self-test passed")
