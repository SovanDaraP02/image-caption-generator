"""
Vocabulary: maps words <-> integer indices, with special tokens for
padding, unknown words, and sequence start/end.

Day 3 concept check: why do we need <start>/<end>/<pad>/<unk>?
- <start> / <end>: tell the decoder where a caption begins and stops,
  so at inference time it knows when to quit generating words.
- <pad>: captions in a batch have different lengths; we pad the short
  ones so they can all sit in one tensor. The loss function is told to
  ignore <pad> positions (see train.py, CrossEntropyLoss(ignore_index=...)).
- <unk>: any word that appears too rarely in training gets mapped here,
  so a handful of one-off words in Flickr8k don't blow up your vocab size.
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
        self.word2idx = {}
        self.idx2word = {}
        self._build_special_tokens()

    def _build_special_tokens(self):
        for token in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]:
            self._add_word(token)

    def _add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    @staticmethod
    def tokenize(caption: str):
        """Lowercase + strip punctuation + split on whitespace.
        Simple and dependency-free; swap for nltk.word_tokenize if you
        want slightly better handling of contractions etc."""
        caption = caption.lower()
        caption = re.sub(r"[^a-z0-9\s]", "", caption)
        return caption.split()

    def build(self, captions: list[str]):
        """captions: list of raw caption strings (training split only —
        never build the vocabulary using validation/test captions, that's
        a data leak)."""
        counter = Counter()
        for cap in captions:
            counter.update(self.tokenize(cap))

        for word, freq in counter.items():
            if freq >= self.min_word_freq:
                self._add_word(word)

        return self

    def encode(self, caption: str, max_len: int = None):
        """Turn a raw caption string into a list of token indices,
        wrapped with <start> ... <end>."""
        tokens = self.tokenize(caption)
        ids = [self.word2idx[self.START_TOKEN]]
        ids += [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]
        ids.append(self.word2idx[self.END_TOKEN])

        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True):
        words = [self.idx2word.get(i, self.UNK_TOKEN) for i in ids]
        if strip_special:
            words = [w for w in words if w not in
                     (self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN)]
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


if __name__ == "__main__":
    # quick self-test — run with: python vocabulary.py
    sample_captions = [
        "A brown dog running across a grassy field",
        "A dog runs through the grass",
        "Two dogs playing in a field of grass",
        "A brown dog running across a grassy field",  # repeat to pass freq threshold
        "A brown dog running across a grassy field",
        "A brown dog running across a grassy field",
        "A brown dog running across a grassy field",
    ]
    vocab = Vocabulary(min_word_freq=2).build(sample_captions)
    print(f"Vocab size: {len(vocab)}")
    encoded = vocab.encode("A brown dog running")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {vocab.decode(encoded)}")
    assert vocab.decode(encoded) != "", "decode should not be empty"
    print("vocabulary.py self-test passed")
