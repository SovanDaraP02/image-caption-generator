"""
CaptionModel: composes EncoderCNN + DecoderWithAttention into one module,
plus greedy and beam-search generation for inference (Day 12).

Day 12 concept check — greedy vs. beam search:
- Greedy: at each step, take the single highest-probability word and
  commit to it. Fast, but one bad early pick can derail the whole
  sentence, and it can get stuck repeating itself.
- Beam search: keep the top-k partial sequences at every step instead of
  just one, and only commit at the end. Costs k times the compute, but
  usually produces noticeably better captions. Implement both -- being
  able to explain *why* beam search helps is a good interview answer.
"""

import torch
import torch.nn.functional as F

from encoder import EncoderCNN
from decoder import DecoderWithAttention


class CaptionModel:
    """Thin wrapper, not an nn.Module itself -- holds a trained encoder +
    decoder pair and provides inference methods. Training uses encoder
    and decoder directly (see train.py) since that's clearer for
    teacher-forcing forward passes."""

    def __init__(self, encoder: EncoderCNN, decoder: DecoderWithAttention,
                 vocab, device: str = "cpu", max_len: int = 25):
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def generate_greedy(self, image: torch.Tensor):
        """image: (1, 3, 224, 224). Returns (caption_string, alphas list)."""
        encoder_out = self.encoder(image.to(self.device))  # (1, 49, 2048)
        h, c = self.decoder.init_hidden_state(encoder_out)

        word = torch.tensor([self.vocab.word2idx[self.vocab.START_TOKEN]], device=self.device)
        generated_ids = []
        alphas = []

        for _ in range(self.max_len):
            embedding = self.decoder.embedding(word)  # (1, embed_dim)
            context, alpha = self.decoder.attention(encoder_out, h)
            lstm_input = torch.cat([embedding, context], dim=1)
            h, c = self.decoder.lstm_cell(lstm_input, (h, c))
            logits = self.decoder.fc(h)  # (1, vocab_size)

            word = logits.argmax(dim=1)  # greedy pick
            token_id = word.item()
            alphas.append(alpha.squeeze(0).cpu())

            if token_id == self.vocab.word2idx[self.vocab.END_TOKEN]:
                break
            generated_ids.append(token_id)

        return self.vocab.decode(generated_ids), alphas

    @torch.no_grad()
    def generate_beam(self, image: torch.Tensor, beam_width: int = 3):
        """image: (1, 3, 224, 224). Returns the best caption string."""
        encoder_out = self.encoder(image.to(self.device))  # (1, 49, 2048)
        h0, c0 = self.decoder.init_hidden_state(encoder_out)

        start_id = self.vocab.word2idx[self.vocab.START_TOKEN]
        end_id = self.vocab.word2idx[self.vocab.END_TOKEN]

        # each beam: (token_id_sequence, log_prob, h, c)
        beams = [([start_id], 0.0, h0, c0)]
        completed = []

        for _ in range(self.max_len):
            candidates = []
            for seq, log_prob, h, c in beams:
                if seq[-1] == end_id:
                    completed.append((seq, log_prob))
                    continue

                word = torch.tensor([seq[-1]], device=self.device)
                embedding = self.decoder.embedding(word)
                context, _ = self.decoder.attention(encoder_out, h)
                lstm_input = torch.cat([embedding, context], dim=1)
                h_new, c_new = self.decoder.lstm_cell(lstm_input, (h, c))
                logits = self.decoder.fc(h_new)
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)  # (vocab_size,)

                top_log_probs, top_ids = log_probs.topk(beam_width)
                for lp, tid in zip(top_log_probs.tolist(), top_ids.tolist()):
                    candidates.append((seq + [tid], log_prob + lp, h_new, c_new))

            if not candidates:
                break

            # keep only the best `beam_width` partial sequences
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

            if len(completed) >= beam_width:
                break

        completed += [(seq, lp) for seq, lp, _, _ in beams]
        best_seq, _ = max(completed, key=lambda x: x[1])
        ids = [t for t in best_seq if t not in (start_id, end_id)]
        return self.vocab.decode(ids)


if __name__ == "__main__":
    # end-to-end self-test with a fully untrained model — this only
    # proves the plumbing works (shapes, control flow), not caption
    # quality, which requires real training on Colab.
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    from vocabulary import Vocabulary

    vocab = Vocabulary(min_word_freq=1).build([
        "a dog runs in the grass", "a cat sits on the mat"
    ])

    encoder = EncoderCNN(fine_tune=False)
    decoder = DecoderWithAttention(vocab_size=len(vocab), encoder_dim=2048)
    model = CaptionModel(encoder, decoder, vocab, device="cpu", max_len=10)

    dummy_image = torch.randn(1, 3, 224, 224)

    caption_greedy, alphas = model.generate_greedy(dummy_image)
    print(f"Greedy caption (untrained, expect gibberish): '{caption_greedy}'")
    print(f"Number of attention maps captured: {len(alphas)}, each shape: {alphas[0].shape if alphas else None}")

    caption_beam = model.generate_beam(dummy_image, beam_width=3)
    print(f"Beam-search caption (untrained, expect gibberish): '{caption_beam}'")

    print("caption_model.py self-test passed -- inference pipeline runs end to end")
