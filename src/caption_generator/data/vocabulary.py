"""Vocabulary: maps words to integer indices, with special tokens for
padding, unknown words, and sequence start/end.

<pad> lets variable-length captions share a batch tensor (see
collate_fn in dataset.py; the loss ignores <pad> positions via
CrossEntropyLoss(ignore_index=...) in train.py). <unk> caps vocabulary
size by mapping rare words below min_word_freq to a single token.
<start>/<end> mark sequence boundaries so the decoder knows when to
stop generating at inference time.
"""

import re
from collections import Counter


class Vocabulary:
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_word_freq: int = 5):
        self.min_word_freq = min_word_freq
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._build_special_tokens()

    def _build_special_tokens(self) -> None:
        for token in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]:
            self._add_word(token)

    def _add_word(self, word: str) -> None:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    @staticmethod
    def tokenize(caption: str) -> list[str]:
        """Lowercase + strip punctuation + split on whitespace.

        Simple and dependency-free; swap for nltk.word_tokenize for
        better handling of contractions if that becomes a bottleneck.
        """
        caption = caption.lower()
        caption = re.sub(r"[^a-z0-9\s]", "", caption)
        return caption.split()

    def build(self, captions: list[str]) -> "Vocabulary":
        """Fit the vocabulary on raw caption strings.

        Only pass the training split -- fitting on validation/test
        captions is a data leak.
        """
        counter: Counter[str] = Counter()
        for cap in captions:
            counter.update(self.tokenize(cap))

        for word, freq in counter.items():
            if freq >= self.min_word_freq:
                self._add_word(word)

        return self

    def encode(self, caption: str, max_len: int | None = None) -> list[int]:
        """Turn a raw caption string into token indices, wrapped with
        <start> ... <end>."""
        tokens = self.tokenize(caption)
        ids = [self.word2idx[self.START_TOKEN]]
        ids += [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]
        ids.append(self.word2idx[self.END_TOKEN])

        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        words = [self.idx2word.get(i, self.UNK_TOKEN) for i in ids]
        if strip_special:
            words = [w for w in words if w not in (self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN)]
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)
