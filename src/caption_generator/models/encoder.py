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

    def __init__(self, fine_tune: bool = False, pretrained: bool = True, backbone: str = "resnet50"):
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
                print(
                    f"[EncoderCNN] Could not download pretrained weights ({e}). "
                    f"Falling back to random init -- fine for shape tests, "
                    f"NOT fine for real training."
                )
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
        features = self.resnet(images)  # (B, 2048, 7, 7)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # (B, 7, 7, 2048)
        features = features.reshape(B, H * W, C)  # (B, 49, 2048)
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


class EncoderCLIP(nn.Module):
    """Wraps a pretrained CLIP vision tower instead of an ImageNet-classifier
    backbone.

    Why this instead of another EncoderCNN backbone: ResNet's ImageNet
    pretraining objective is 1000-way object classification, which has no
    linguistic structure to it at all -- "this is a golden retriever" and
    "this is a beagle" are just two different output classes, equally
    unrelated to every other class. CLIP is pretrained contrastively to
    align images with their natural-language captions directly, so its
    features already encode the kind of visual-semantic structure a
    captioning decoder needs, instead of the decoder having to learn that
    structure from scratch on top of classification features that were
    never built for it. This is the encoder-side counterpart to the
    dataset-scale finding in README.md (6k->50k->113k images): more caption
    data stopped helping much once the frozen ResNet features became the
    bottleneck instead of the data.

    Uses openai/clip-vit-base-patch32: a 224x224 image split into 32x32
    patches gives exactly (224/32)^2 = 49 patch tokens, matching
    EncoderCNN's 49-region grid with no interpolation needed -- so this is
    shape-compatible everywhere EncoderCNN is used, as long as the caller
    also passes the right encoder_dim (768, not EncoderCNN's 2048) to
    DecoderWithAttention/Attention, and normalizes inputs with CLIP_MEAN/
    CLIP_STD (see data/dataset.py) rather than ImageNet stats.
    """

    OUTPUT_DIM = 768
    CHECKPOINT = "openai/clip-vit-base-patch32"

    def __init__(self, fine_tune: bool = False, pretrained: bool = True):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModel

        if pretrained:
            try:
                self.clip = CLIPVisionModel.from_pretrained(self.CHECKPOINT)
            except Exception as e:
                print(
                    f"[EncoderCLIP] Could not download pretrained weights ({e}). "
                    f"Falling back to random init -- fine for shape tests, "
                    f"NOT fine for real training."
                )
                self.clip = CLIPVisionModel(CLIPVisionConfig(image_size=224, patch_size=32))
        else:
            self.clip = CLIPVisionModel(CLIPVisionConfig(image_size=224, patch_size=32))

        self.fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 224, 224), normalized with CLIP_MEAN/CLIP_STD
        returns: (B, 49, 768) -- 49 spatial patches, each 768-dim
        """
        out = self.clip(pixel_values=images).last_hidden_state  # (B, 50, 768): CLS + 49 patches
        return out[:, 1:, :]  # drop the CLS token -> (B, 49, 768)

    def fine_tune(self, fine_tune: bool = False) -> None:
        """Freeze everything by default. If fine-tuning, unfreeze only the
        last transformer block -- same rationale as EncoderCNN.fine_tune."""
        for param in self.clip.parameters():
            param.requires_grad = False
        if fine_tune:
            for layer in self.clip.encoder.layers[-1:]:
                for param in layer.parameters():
                    param.requires_grad = True
