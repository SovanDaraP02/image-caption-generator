"""
DecoderWithAttention: generates the caption one word at a time.

Day 8-9 concept check:
- Embedding: turns a word index into a dense vector the network can do
  math with (a lookup table, learned during training).
- LSTMCell: the recurrent memory. You don't need to derive its gate
  equations by hand -- know that the cell state c_t carries long-term
  information forward, and the hidden state h_t is what gets used for
  both the attention query and the word prediction at this step.
- We concatenate the word embedding with the attention context vector
  z_t before feeding the LSTM -- that's how "what I'm looking at" and
  "what word comes next" get combined at every single timestep.
- Teacher forcing (Day 9): during TRAINING, the input at step t is the
  ground-truth previous word, not the model's own (possibly wrong)
  prediction. This trains faster and more stably, but it's also why
  inference-time generation (Section 4.4 of the blueprint) can behave
  differently from training loss alone -- exposure bias.
"""

import torch
import torch.nn as nn

from attention import Attention


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 encoder_dim: int = 2048, decoder_dim: int = 512,
                 attention_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # LSTMCell input = word embedding concatenated with context vector
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # initialize h_0, c_0 from the mean-pooled image features, rather
        # than starting from zero -- gives the decoder a head start on
        # "what's roughly in this image" before it's said a single word
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)  # vocabulary projection
        self.dropout = nn.Dropout(0.5)

    def init_hidden_state(self, encoder_out: torch.Tensor):
        mean_encoder_out = encoder_out.mean(dim=1)   # (B, encoder_dim)
        h = self.init_h(mean_encoder_out)             # (B, decoder_dim)
        c = self.init_c(mean_encoder_out)              # (B, decoder_dim)
        return h, c

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor):
        """
        Training-mode forward pass with teacher forcing.

        encoder_out: (B, 49, encoder_dim)
        captions:    (B, T) token indices, already <start>-prefixed
        returns:
            logits: (B, T-1, vocab_size)  -- predictions for each position
            alphas: (B, T-1, 49)           -- attention weights at each step,
                                              useful for the regularization
                                              term (blueprint Section 4.2)
                                              and for heatmap visualization
        """
        B, T = captions.shape
        h, c = self.init_hidden_state(encoder_out)

        embeddings = self.embedding(captions)  # (B, T, embed_dim)

        # we predict tokens 1..T-1 from tokens 0..T-2 (teacher forcing),
        # so we run T-1 decoding steps
        num_steps = T - 1
        logits = torch.zeros(B, num_steps, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(B, num_steps, encoder_out.shape[1], device=encoder_out.device)

        for t in range(num_steps):
            context, alpha = self.attention(encoder_out, h)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits[:, t, :] = self.fc(self.dropout(h))
            alphas[:, t, :] = alpha

        return logits, alphas


if __name__ == "__main__":
    # shape self-test — run with: python decoder.py
    B, num_regions, encoder_dim, vocab_size, T = 2, 49, 2048, 50, 7

    decoder = DecoderWithAttention(vocab_size=vocab_size, encoder_dim=encoder_dim)
    dummy_encoder_out = torch.randn(B, num_regions, encoder_dim)
    dummy_captions = torch.randint(0, vocab_size, (B, T))  # fake <start>...<end> sequences

    logits, alphas = decoder(dummy_encoder_out, dummy_captions)

    print(f"logits shape: {tuple(logits.shape)}  (expected: ({B}, {T-1}, {vocab_size}))")
    print(f"alphas shape: {tuple(alphas.shape)}  (expected: ({B}, {T-1}, {num_regions}))")
    assert logits.shape == (B, T - 1, vocab_size)
    assert alphas.shape == (B, T - 1, num_regions)

    # confirm the loss actually computes with no shape errors — the #1
    # beginner bug per the blueprint's debugging checklist (Section 8)
    criterion = nn.CrossEntropyLoss()
    targets = dummy_captions[:, 1:]  # predict tokens 1..T-1
    loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
    print(f"loss computed successfully: {loss.item():.4f}")

    print("decoder.py self-test passed")
