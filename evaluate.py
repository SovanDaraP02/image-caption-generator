"""
Evaluation: run trained model against the test split, score with
BLEU-1..4 / METEOR / CIDEr via pycocoevalcap.

Day 12 concept check — what each metric actually measures:
- BLEU: n-gram precision against reference captions. Cheap to compute,
  widely reported, but purely surface-level (doesn't understand synonyms).
- METEOR: also considers synonyms/stemming, correlates better with human
  judgment than BLEU alone.
- CIDEr: built specifically for image captioning; weights n-grams by
  TF-IDF, so it penalizes generic captions ("a photo of something") more
  than BLEU does.
Report all three -- a model can look good on one and mediocre on another,
and that gap is itself worth discussing in your face-to-face.

Install first:  pip install pycocoevalcap
"""

import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "data")
sys.path.insert(0, "models")
from vocabulary import Vocabulary            # noqa: E402
from dataset import Flickr8kDataset, collate_fn  # noqa: E402
from encoder import EncoderCNN                # noqa: E402
from decoder import DecoderWithAttention      # noqa: E402
from caption_model import CaptionModel        # noqa: E402


def evaluate(checkpoint_path: str, test_pairs_by_image: dict, image_dir: str, device: str = "cpu"):
    """
    test_pairs_by_image: dict of {image_filename: [ref_caption_1, ref_caption_2, ...]}
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

        gts[i] = references
        res[i] = [generated_caption]

    # pycocoevalcap wants this exact import path
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.cider.cider import Cider

    scores = {}
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)
    for n, score in enumerate(bleu_score, start=1):
        scores[f"BLEU-{n}"] = score

    meteor_score, _ = Meteor().compute_score(gts, res)
    scores["METEOR"] = meteor_score

    cider_score, _ = Cider().compute_score(gts, res)
    scores["CIDEr"] = cider_score

    return scores


if __name__ == "__main__":
    print("This script needs a real trained checkpoint (best_checkpoint.pth from train.py)")
    print("and the test split's reference captions to run for real.")
    print("Structure check only:")
    print("  evaluate(checkpoint_path, test_pairs_by_image, image_dir) -> ")
    print("  {'BLEU-1': .., 'BLEU-2': .., 'BLEU-3': .., 'BLEU-4': .., 'METEOR': .., 'CIDEr': ..}")
