# What I learned building an image captioning model from scratch

I built a neural network that looks at a photo and writes a caption for it — encoder, attention mechanism, and decoder, all hand-implemented in PyTorch, no pretrained captioning model involved. The interesting part wasn't getting it to run. It was what happened when I tried to make it *better*, four times in a row, and had to figure out why each attempt did or didn't work.

**Attempt 1 → 2: more data.** Baseline was Flickr8k, 6,000 images. Scaling to a 50,000-image COCO subset gave a real jump: BLEU-4 up 38%, CIDEr up 48%. More examples, same everything else, meaningfully better model.

**Attempt 2 → 3: more data again.** I more than doubled the training set again, to 113,000 images. BLEU-4 moved less than 1%. Validation loss visibly flattened in the last few epochs. Same lever, pulled the same way — and this time it barely moved.

That's the moment worth pausing on. It would've been easy to conclude "we've plateaued, this architecture has a ceiling." Instead I asked a more specific question: is the bottleneck *data volume*, or is it something the extra data can't fix?

The encoder was a ResNet-50, pretrained on ImageNet classification — 1,000 object labels, no linguistic structure at all. "Golden retriever" and "beagle" are just two unrelated output classes to it. The decoder was trying to turn those labels into sentences. Maybe the features themselves were the ceiling, not the dataset size.

**Attempt 4: same 113k images, swap the encoder for CLIP** — pretrained not on classification but on matching images to their actual captions. Same data, same decoder, one variable changed. BLEU-4 jumped 12.4% — bigger than the entire second data-scaling attempt had delivered. The hypothesis held.

**Attempt 5: fine-tune that CLIP encoder's last layer**, at a learning rate 10x lower than the decoder's, warm-started from the converged checkpoint. Another real gain (+4.5% BLEU-4), and validation loss showed early signs of its own plateau by the final epoch — consistent, not a fluke.

Final result: BLEU-4 0.26, CIDEr 0.81 on a 5,000-image held-out COCO test split, up from BLEU-4 0.16 at the Flickr8k baseline. The bigger takeaway wasn't the final number — it was that "add more data" and "fix the architecture" are different fixes for different problems, and the only way to know which one you have is to isolate the variable and measure it.

Full writeup, code, and all five checkpoints compared: see the [README](./README.md).
