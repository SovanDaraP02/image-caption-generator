from vocabulary import Vocabulary


def test_build_assigns_special_tokens_first():
    vocab = Vocabulary(min_word_freq=1).build(["a brown dog running"])
    assert vocab.word2idx[vocab.PAD_TOKEN] == 0
    assert vocab.word2idx[vocab.START_TOKEN] == 1
    assert vocab.word2idx[vocab.END_TOKEN] == 2
    assert vocab.word2idx[vocab.UNK_TOKEN] == 3


def test_min_word_freq_drops_rare_words():
    vocab = Vocabulary(min_word_freq=2).build([
        "a dog runs", "a dog barks", "a rare gizmo appears",
    ])
    assert "dog" in vocab.word2idx
    assert "gizmo" not in vocab.word2idx


def test_encode_wraps_with_start_and_end():
    vocab = Vocabulary(min_word_freq=1).build(["a brown dog running"])
    ids = vocab.encode("a brown dog running")
    assert ids[0] == vocab.word2idx[vocab.START_TOKEN]
    assert ids[-1] == vocab.word2idx[vocab.END_TOKEN]


def test_encode_unknown_word_maps_to_unk():
    vocab = Vocabulary(min_word_freq=1).build(["a brown dog running"])
    ids = vocab.encode("a spaceship")
    unk_id = vocab.word2idx[vocab.UNK_TOKEN]
    assert unk_id in ids


def test_decode_roundtrip_drops_special_tokens():
    vocab = Vocabulary(min_word_freq=1).build(["a brown dog running"])
    ids = vocab.encode("a brown dog running")
    decoded = vocab.decode(ids[1:-1])  # strip start/end like generation code does
    assert decoded == "a brown dog running"
