"""DecoderWithAttention: generates the caption one word at a time.

At each timestep, the word embedding is concatenated with the
attention context vector before entering the LSTM -- this is how
"what I'm looking at" and "what word comes next" are combined at every
step. h_0/c_0 are initialized from the mean-pooled image features
rather than zero, giving the decoder a head start before it has said
a single word.

Training uses teacher forcing: the input at step t is the ground-truth
previous token, not the model's own prediction. This trains faster and
more stably, but it's also why inference-time generation quality can
diverge from training loss alone (exposure bias) -- see
caption_model.py for the greedy/beam-search inference paths that don't
have this crutch.
"""

import torch
import torch.nn as nn

from caption_generator.models.attention import Attention


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

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)  # vocabulary projection
        self.dropout = nn.Dropout(0.5)

    def init_hidden_state(self, encoder_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_encoder_out = encoder_out.mean(dim=1)   # (B, encoder_dim)
        h = self.init_h(mean_encoder_out)             # (B, decoder_dim)
        c = self.init_c(mean_encoder_out)              # (B, decoder_dim)
        return h, c

    def forward(self, encoder_out: torch.Tensor,
                captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training-mode forward pass with teacher forcing.

        encoder_out: (B, 49, encoder_dim)
        captions:    (B, T) token indices, already <start>-prefixed
        returns:
            logits: (B, T-1, vocab_size)  -- predictions for each position
            alphas: (B, T-1, 49)           -- attention weights at each step,
                                              used by the attention
                                              regularization term in
                                              train.py and for heatmap
                                              visualization
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
