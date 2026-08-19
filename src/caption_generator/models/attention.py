"""Bahdanau (additive) attention over spatial image regions.

At each decoding step, scores how relevant each of the 49 encoder
regions is to the decoder's current hidden state, normalizes those
scores with softmax, and returns a weighted summary of the regions
(the context vector) plus the weights themselves (used both as the
LSTM's per-step input and for the attention heatmaps in
visualize_attention.py).

    energy_i = w^T tanh(W_a * region_i + W_h * h_{t-1})   -- (B, 49)
    alpha    = softmax(energy)                              -- (B, 49), sums to 1
    context  = sum_i(alpha_i * region_i)                    -- (B, encoder_dim)
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_dim: int = 2048, decoder_dim: int = 512, attention_dim: int = 256):
        super().__init__()
        self.W_a = nn.Linear(encoder_dim, attention_dim)  # projects image features
        self.W_h = nn.Linear(decoder_dim, attention_dim)  # projects decoder hidden state
        self.w = nn.Linear(attention_dim, 1)  # collapses to one score per region
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        encoder_out:    (B, 49, encoder_dim)   -- the 49 image regions
        decoder_hidden: (B, decoder_dim)        -- h_{t-1}
        returns:
            context: (B, encoder_dim)  -- weighted image summary
            alpha:   (B, 49)            -- attention weights, sum to 1 per image
        """
        att1 = self.W_a(encoder_out)  # (B, 49, attention_dim)
        att2 = self.W_h(decoder_hidden).unsqueeze(1)  # (B, 1, attention_dim)
        energy = self.w(self.tanh(att1 + att2)).squeeze(2)  # (B, 49)

        alpha = self.softmax(energy)  # (B, 49), sums to 1 per image
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)

        return context, alpha
