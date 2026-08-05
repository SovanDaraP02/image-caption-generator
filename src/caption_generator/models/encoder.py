"""EncoderCNN: wraps a pretrained ResNet backbone, strips the
classification head, and returns spatial feature maps instead of a
single pooled vector.

The spatial grid (7x7 = 49 regions) is kept rather than global-average-
pooling to one vector, because the attention mechanism needs to choose
*where* to look for each word -- a pooled vector has already discarded
that spatial information.
"""

import torch
import torch.nn as nn
import torchvision


class EncoderCNN(nn.Module):
    # All three are drop-in swaps: same 2048-channel output, so nothing
    # downstream (attention_dim, decoder_dim, etc.) needs to change.
    _BACKBONES = {
        "resnet50": (torchvision.models.resnet50, "ResNet50_Weights", "IMAGENET1K_V2"),
        "resnet101": (torchvision.models.resnet101, "ResNet101_Weights", "IMAGENET1K_V2"),
        "resnet152": (torchvision.models.resnet152, "ResNet152_Weights", "IMAGENET1K_V2"),
    }

    def __init__(self, fine_tune: bool = False, pretrained: bool = True,
                 backbone: str = "resnet50"):
        super().__init__()

        if backbone not in self._BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}', choose from {list(self._BACKBONES)}")
        model_fn, weights_enum_name, weights_tag = self._BACKBONES[backbone]

        # Pretrained ImageNet weights require network access to
        # download.pytorch.org. Falls back to random init so this module
        # stays importable/testable offline; training quality depends on
        # actually getting the pretrained weights, so check the printed
        # message below. Tests pass pretrained=False to skip the network
        # call entirely and stay fast/deterministic in CI.
        if pretrained:
            try:
                weights_enum = getattr(torchvision.models, weights_enum_name)
                resnet = model_fn(weights=getattr(weights_enum, weights_tag))
            except Exception as e:
                print(f"[EncoderCNN] Could not download pretrained weights ({e}). "
                      f"Falling back to random init -- fine for shape tests, "
                      f"NOT fine for real training.")
                resnet = model_fn(weights=None)
        else:
            resnet = model_fn(weights=None)

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

    def fine_tune(self, fine_tune: bool = False) -> None:
        """Freeze everything by default. If fine-tuning, unfreeze only
        the last residual block -- unfreezing the whole network on a
        dataset this small causes catastrophic forgetting of the
        pretrained features."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        if fine_tune:
            for layer in list(self.resnet.children())[7:]:
                for param in layer.parameters():
                    param.requires_grad = True
