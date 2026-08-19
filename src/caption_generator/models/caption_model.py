"""CaptionModel: composes EncoderCNN + DecoderWithAttention into one
unit, plus greedy and beam-search inference.

Greedy decoding commits to the single highest-probability word at each
step -- fast, but one bad early pick can derail the rest of the
sentence, and it's prone to repetition loops. Beam search keeps the
top-k partial sequences at every step and only commits at the end,
trading k times the compute for noticeably better captions.
"""

import torch
import torch.nn.functional as F

from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCLIP, EncoderCNN


def _would_repeat_ngram(seq: list[int], next_id: int, n: int) -> bool:
    """True if appending next_id to seq would create an n-gram that
    already occurred earlier in seq. Standard decoding-time repetition
    guard (e.g. HuggingFace's `no_repeat_ngram_size`) -- it blocks loops
    like "white dress and a white dress" without needing retraining,
    since the underlying cause (a lightly-trained decoder falling back
    to a familiar phrase when uncertain) is a training-data/capacity
    issue that no amount of decoding cleverness fully fixes, but n-gram
    blocking prevents the most visible symptom of it."""
    if n <= 0 or len(seq) < n - 1:
        return False
    candidate = tuple(seq[-(n - 1) :] + [next_id])
    return any(tuple(seq[i : i + n]) == candidate for i in range(len(seq) - n + 1))


class CaptionModel:
    """Thin wrapper, not an nn.Module itself -- holds a trained encoder +
    decoder pair and provides inference methods. Training uses encoder
    and decoder directly (see train.py) since that's clearer for
    teacher-forcing forward passes."""

    def __init__(
        self,
        encoder: EncoderCNN | EncoderCLIP,
        decoder: DecoderWithAttention,
        vocab: Vocabulary,
        device: str = "cpu",
        max_len: int = 25,
    ):
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def generate_greedy(
        self, image: torch.Tensor, no_repeat_ngram_size: int = 3
    ) -> tuple[str, list[torch.Tensor]]:
        """image: (1, 3, 224, 224). Returns (caption_string, alphas list).

        no_repeat_ngram_size: block picking a token that would recreate
        an n-gram already generated (0 disables this).
        """
        encoder_out = self.encoder(image.to(self.device))  # (1, 49, 2048)
        h, c = self.decoder.init_hidden_state(encoder_out)

        word = torch.tensor([self.vocab.word2idx[self.vocab.START_TOKEN]], device=self.device)
        generated_ids: list[int] = []
        alphas = []
        unk_id = self.vocab.word2idx[self.vocab.UNK_TOKEN]

        for _ in range(self.max_len):
            embedding = self.decoder.embedding(word)  # (1, embed_dim)
            context, alpha = self.decoder.attention(encoder_out, h)
            lstm_input = torch.cat([embedding, context], dim=1)
            h, c = self.decoder.lstm_cell(lstm_input, (h, c))
            logits = self.decoder.fc(h)  # (1, vocab_size)

            # walk candidates best-to-worst, skip <unk> (never a real word to
            # say -- forces the model to commit to its next-best actual
            # vocabulary word) and anything that would create a repeated
            # n-gram; fall back to the top non-<unk> pick if every candidate
            # would repeat an n-gram (never falls back to <unk> itself)
            ranked_ids = logits.squeeze(0).argsort(descending=True).tolist()
            token_id = next((i for i in ranked_ids if i != unk_id), ranked_ids[0])
            for candidate_id in ranked_ids:
                if candidate_id == unk_id:
                    continue
                if not _would_repeat_ngram(generated_ids, candidate_id, no_repeat_ngram_size):
                    token_id = candidate_id
                    break
            word = torch.tensor([token_id], device=self.device)
            alphas.append(alpha.squeeze(0).cpu())

            if token_id == self.vocab.word2idx[self.vocab.END_TOKEN]:
                break
            generated_ids.append(token_id)

        return self.vocab.decode(generated_ids), alphas

    @torch.no_grad()
    def generate_beam(self, image: torch.Tensor, beam_width: int = 3, no_repeat_ngram_size: int = 3) -> str:
        """image: (1, 3, 224, 224). Returns the best caption string.

        no_repeat_ngram_size: block candidates that would recreate an
        n-gram already generated in that beam (0 disables this).
        """
        encoder_out = self.encoder(image.to(self.device))  # (1, 49, 2048)
        h0, c0 = self.decoder.init_hidden_state(encoder_out)

        start_id = self.vocab.word2idx[self.vocab.START_TOKEN]
        end_id = self.vocab.word2idx[self.vocab.END_TOKEN]
        unk_id = self.vocab.word2idx[self.vocab.UNK_TOKEN]

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

                # over-fetch so there's still `beam_width` options left
                # after any get filtered out by the <unk>/n-gram checks
                top_log_probs, top_ids = log_probs.topk(min(beam_width * 4, log_probs.shape[0]))
                kept = 0
                for lp, tid in zip(top_log_probs.tolist(), top_ids.tolist(), strict=True):
                    if tid == unk_id:
                        continue
                    if _would_repeat_ngram(seq, tid, no_repeat_ngram_size):
                        continue
                    candidates.append((seq + [tid], log_prob + lp, h_new, c_new))
                    kept += 1
                    if kept >= beam_width:
                        break

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
