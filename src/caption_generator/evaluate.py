"""Evaluation: run a trained checkpoint against the test split, score
with BLEU-1..4 / METEOR / CIDEr via pycocoevalcap.

Report all three rather than just BLEU: BLEU is n-gram precision only
(cheap, widely reported, but surface-level -- no synonym awareness).
METEOR adds synonym/stemming matching and correlates better with human
judgment. CIDEr is purpose-built for captioning -- it TF-IDF-weights
n-grams, so it penalizes generic captions ("a photo of something") more
than BLEU does. A model can look good on one metric and mediocre on
another; that gap is informative, not noise.

Install first:  pip install pycocoevalcap
"""

import torch

from caption_generator.data.vocabulary import Vocabulary
from caption_generator.models.caption_model import CaptionModel
from caption_generator.models.decoder import DecoderWithAttention
from caption_generator.models.encoder import EncoderCNN


def evaluate(checkpoint_path: str, test_pairs_by_image: dict[str, list[str]],
             image_dir: str, device: str = "cpu") -> dict[str, float]:
    """
    test_pairs_by_image: {image_filename: [ref_caption_1, ref_caption_2, ...]}
                          -- captioning metrics need ALL reference captions
                          per image, not just one, since Flickr8k has 5
                          human-written captions per image.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vocab = Vocabulary()
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = checkpoint["vocab_idx2word"]

    encoder = EncoderCNN(fine_tune=False)
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder = DecoderWithAttention(vocab_size=len(vocab))
    decoder.load_state_dict(checkpoint["decoder_state"])

    model = CaptionModel(encoder, decoder, vocab, device=device)

    import torchvision.transforms as T
    from PIL import Image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    gts, res = {}, {}  # ground truths / results, keyed by image id, pycocoevalcap format
    for i, (filename, references) in enumerate(test_pairs_by_image.items()):
        image = Image.open(f"{image_dir}/{filename}").convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        generated_caption, _ = model.generate_greedy(image_tensor)
        # An empty candidate (an undertrained model emitting <end> as its
        # first token) desyncs pycocoevalcap's METEOR scorer -- it shells
        # out to a Java process and reads exactly one score line per
        # image, so a missing line there throws every later readline()
        # off by one, surfacing as a ValueError far from its real cause.
        # A harmless placeholder keeps line counts aligned.
        if not generated_caption.strip():
            generated_caption = "<empty>"

        gts[i] = references
        res[i] = [generated_caption]

    # pycocoevalcap wants this exact import path
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor

    scores: dict[str, float] = {}

    bleu_score, _ = Bleu(4).compute_score(gts, res)
    for n, score in enumerate(bleu_score, start=1):
        scores[f"BLEU-{n}"] = score

    # METEOR shells out to a Java subprocess; a version/environment
    # mismatch there is a known source of parsing failures unrelated to
    # the model itself. Don't let it take BLEU/CIDEr down with it.
    try:
        meteor_score, _ = Meteor().compute_score(gts, res)
        scores["METEOR"] = meteor_score
    except Exception as e:
        print(f"METEOR scoring failed ({e}); reporting BLEU/CIDEr only.")

    cider_score, _ = Cider().compute_score(gts, res)
    scores["CIDEr"] = cider_score

    return scores


if __name__ == "__main__":
    print("This script needs a real trained checkpoint (best_checkpoint.pth from train.py)")
    print("and the test split's reference captions to run for real.")
    print("Structure check only:")
    print("  evaluate(checkpoint_path, test_pairs_by_image, image_dir) -> ")
    print("  {'BLEU-1': .., 'BLEU-2': .., 'BLEU-3': .., 'BLEU-4': .., 'METEOR': .., 'CIDEr': ..}")
