"""
Attention: at each word-generation step, decide how much to "look at"
each of the 49 image regions, based on what the decoder has generated
so far (its previous hidden state).

Day 6-7 concept check — walk through this on paper with tiny numbers
before trusting the code:
  1. energy score e_ti = w^T tanh(W_a * a_i + W_h * h_{t-1})
     -> one scalar "how relevant is region i to what I'm about to say"
        for each of the 49 regions
  2. alpha_ti = softmax(e_t)
     -> normalize the 49 scores so they sum to 1 (a probability
        distribution over image regions)
  3. context vector z_t = sum_i (alpha_ti * a_i)
     -> a weighted average of the 49 region vectors, weighted by how
        relevant each one is right now
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_dim: int = 2048, decoder_dim: int = 512, attention_dim: int = 256):
        super().__init__()
        self.W_a = nn.Linear(encoder_dim, attention_dim)   # projects image features
        self.W_h = nn.Linear(decoder_dim, attention_dim)   # projects decoder hidden state
        self.w = nn.Linear(attention_dim, 1)                # collapses to one score per region
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        encoder_out:    (B, 49, encoder_dim)   -- the 49 image regions
        decoder_hidden: (B, decoder_dim)        -- h_{t-1}
        returns:
            context: (B, encoder_dim)  -- z_t, the weighted image summary
            alpha:   (B, 49)            -- the attention weights themselves
                                            (this is what you visualize as
                                            a heatmap in Section 9/13 of the plan)
        """
        att1 = self.W_a(encoder_out)                       # (B, 49, attention_dim)
        att2 = self.W_h(decoder_hidden).unsqueeze(1)        # (B, 1, attention_dim)
        energy = self.w(self.tanh(att1 + att2)).squeeze(2)  # (B, 49)

        alpha = self.softmax(energy)                        # (B, 49), sums to 1 per image
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)

        return context, alpha


if __name__ == "__main__":
    # shape + sanity self-test — run with: python attention.py
    B, num_regions, encoder_dim, decoder_dim = 2, 49, 2048, 512

    attention = Attention(encoder_dim=encoder_dim, decoder_dim=decoder_dim)
    dummy_encoder_out = torch.randn(B, num_regions, encoder_dim)
    dummy_hidden = torch.randn(B, decoder_dim)

    context, alpha = attention(dummy_encoder_out, dummy_hidden)

    print(f"context shape: {tuple(context.shape)}  (expected: ({B}, {encoder_dim}))")
    print(f"alpha shape:   {tuple(alpha.shape)}  (expected: ({B}, {num_regions}))")
    assert context.shape == (B, encoder_dim)
    assert alpha.shape == (B, num_regions)

    # the whole point of softmax: weights must sum to 1 for each image
    sums = alpha.sum(dim=1)
    print(f"alpha row sums: {sums.tolist()}  (expected: ~1.0 each)")
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), "attention weights must sum to 1"

    print("attention.py self-test passed")
